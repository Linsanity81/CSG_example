//
// Created by ziqwang on 30.11.21.
//

#include "MeshVoxelOpt.h"
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/edges.h>
#include <set>
#include <iostream>

void MeshVoxelOpt::readMesh(std::string filename, std::string tetgen_str = "pq1.414Y") {

    MeshVoxel::readMesh(filename);
    igl::copyleft::tetgen::tetrahedralize(meshV_, meshF_, tetgen_str, TV_, TT_, TF_);
    computeEdgeLength();
}

void MeshVoxelOpt::computeEdgeLength(){
    igl::edges(TT_, E_);
    l0.clear();
    for(int id = 0; id < E_.rows(); id++){
        Eigen::Vector3d pA = TV_.row(E_(id, 0));
        Eigen::Vector3d pB = TV_.row(E_(id, 1));
        l0.push_back((pA - pB).norm());
    }
}

double MeshVoxelOpt::computeTetVolume(int tetID){
    Eigen::Vector3d pA = TV_.row(TT_(tetID, 0));
    Eigen::Vector3d pB = TV_.row(TT_(tetID, 1));
    Eigen::Vector3d pC = TV_.row(TT_(tetID, 2));
    Eigen::Vector3d pD = TV_.row(TT_(tetID, 3));
    return std::abs(((pB - pA).cross(pC - pA)).dot(pD - pA)) / 6.0;
}

void MeshVoxelOpt::approxVoxelization(vector<Eigen::MatrixXd> &Vs,
                                      vector<Eigen::MatrixXi> &Fs,
                                      vector<double> &volumes,
                                      vector<Eigen::Vector3i> &voxel_indices){

    int num_of_voxels = grids_size_ * grids_size_ * grids_size_;
    volumes.resize(num_of_voxels);
    vector<vector<Eigen::Vector3d>> tet_vers(num_of_voxels);
    for(int id = 0; id < TT_.rows(); id++)
    {
        std::set<int> voxel_indices_set;
        vector<Eigen::Vector3d> points;
        for(int jd = 0; jd < 4; jd++){
            Eigen::Vector3d pt = TV_.row(TT_(id, jd));
            Eigen::Vector3i voxel_index = point_to_voxel_index(pt);
            int voxel_digit = index_to_digit(voxel_index);
            voxel_indices_set.insert(voxel_digit);
            points.push_back(pt);
        }

        double tet_volume = computeTetVolume(id);
        if(voxel_indices_set.size() == 1){
            int voxel_index = *voxel_indices_set.begin();
            volumes[voxel_index] += tet_volume;
            tet_vers[voxel_index].insert(tet_vers[voxel_index].end(), points.begin(), points.end());
        }
//        for(auto it = voxel_indices_set.begin(); it != voxel_indices_set.end(); it++){
//            int voxel_index = *it;
//            volumes[voxel_index] += tet_volume;
//            //tet_vers[voxel_index].insert(tet_vers[voxel_index].end(), points.begin(), points.end());
//        }
    }

    double nonempty_minimum_volume_requirement = 1E-6;
    vector<double> volumes_tmp;
    vector<vector<Eigen::Vector3d>> tet_vers_tmp;

    for(int id = 0; id < num_of_voxels; id++){
        Eigen::Vector3i index = digit_to_index(id);
        if(volumes[id] > nonempty_minimum_volume_requirement){
            voxel_indices.push_back(index);
            volumes_tmp.push_back(volumes[id]);
            tet_vers_tmp.push_back(tet_vers[id]);
        }
    }
    volumes = volumes_tmp;
    tet_vers = tet_vers_tmp;

    Vs.resize(tet_vers.size());
    Fs.resize(tet_vers.size());
    for(int id = 0; id < tet_vers.size(); id++){
        Vs[id] = Eigen::MatrixXd(tet_vers[id].size(), 3);
        for(int jd = 0; jd < tet_vers[id].size(); jd++){
            Vs[id].row(jd) = tet_vers[id][jd];
        }

        Fs[id] = Eigen::MatrixXi(tet_vers[id].size(), 3);
        for(int jd = 0; jd < tet_vers[id].size() / 4; jd++){
            Fs[id].row(jd * 4) = Eigen::Vector3i(4 * jd, 4 * jd + 2, 4 * jd + 1);
            Fs[id].row(jd * 4 + 1) = Eigen::Vector3i(4 * jd, 4 * jd + 3, 4 * jd + 2);
            Fs[id].row(jd * 4 + 2) = Eigen::Vector3i(4 * jd, 4 * jd + 1, 4 * jd + 3);
            Fs[id].row(jd * 4 + 3) = Eigen::Vector3i(4 * jd + 1, 4 * jd + 2, 4 * jd + 3);
        }
    }
}

void MeshVoxelOpt::computeSelectedVoxels(vector<double> &selected_voxel_volumes,
                                         vector<Eigen::Vector3i> &selected_voxel_indices)
{
    vector<Eigen::MatrixXd> Vs;
    vector<Eigen::MatrixXi> Fs;
    vector<double> volumes;
    vector<Eigen::Vector3i> voxel_indices;

    approxVoxelization(Vs, Fs, volumes, voxel_indices);
    for(int id = 0; id < volumes.size(); id++){
        if(volumes[id] > minimum_volume_){
            selected_voxel_indices.push_back(voxel_indices[id]);
            selected_voxel_volumes.push_back(volumes[id]);
        }
    }

    return;
}

Eigen::Vector3i MeshVoxelOpt::point_to_voxel_index(Eigen::Vector3d pt) {
    int nx = std::floor((pt[0] - grids_origin_[0]) / grids_width_);
    int ny = std::floor((pt[1] - grids_origin_[1]) / grids_width_);
    int nz = std::floor((pt[2] - grids_origin_[2]) / grids_width_);
    return Eigen::Vector3i(nx, ny, nz);
}

int MeshVoxelOpt::computeDistanceVoxelToVoxel(Eigen::Vector3i voxelA, Eigen::Vector3i voxelB){
    Eigen::Vector3i distance = (voxelA - voxelB).cwiseAbs();
    int minimum_distance = 0;
    for(int kd = 0; kd < 3; kd++){
        minimum_distance += distance[kd] > 0 ? pow(distance[kd] - 1, 2): 0;
    }
    return minimum_distance;
}

void MeshVoxelOpt::computeDiffDistancePointToVoxel(Eigen::Vector3d pt,
                                                   Eigen::Vector3i voxel_index,
                                                   double &distance,
                                                   Eigen::Vector3d &gradient){
    Eigen::Vector3i pt_voxel_index = point_to_voxel_index(pt);
    Eigen::Vector3i voxel_distance = pt_voxel_index - voxel_index;
    distance = 0;
    gradient = Eigen::Vector3d(0, 0, 0);
    for(int kd = 0; kd < 3; kd++)
    {
        if(voxel_distance[kd] > 0)
        {
            double max_voxel_coord = grids_origin_[kd] + (voxel_index[kd] + 1) * grids_width_;
            distance += pow(pt[kd] - max_voxel_coord, 2.0);
            gradient[kd] = 2 * (pt[kd] - max_voxel_coord);
        }
        else if(voxel_distance[kd] < 0){
            double min_voxel_coord = grids_origin_[kd] + voxel_index[kd] * grids_width_;
            distance += pow(min_voxel_coord - pt[kd], 2.0);
            gradient[kd] = 2 * (pt[kd] - min_voxel_coord);
        }
    }
}

void MeshVoxelOpt::computeDiffDistanceToSelectedVoxels(const Eigen::MatrixXd &tv,
                                         const vector<Eigen::Vector3i> &selected_voxel_indices,
                                         double &distance,
                                         Eigen::MatrixXd &gradient){

    std::map<int, bool> map_voxel_selected;
    for(int id = 0; id < selected_voxel_indices.size(); id++)
    {
        Eigen::Vector3i voxel_index = selected_voxel_indices[id];
        int voxel_digit = index_to_digit(voxel_index);
        map_voxel_selected[voxel_digit] = true;
    }

    std::map<int, vector<int>> map_voxel_points;
    for(int id = 0; id < tv.rows(); id++)
    {
        Eigen::Vector3i voxel_index = point_to_voxel_index(tv.row(id));
        int voxel_digit = index_to_digit(voxel_index);
        std::cout << voxel_index.transpose() << std::endl;
        if(!map_voxel_selected[voxel_digit]){
            map_voxel_points[voxel_digit].push_back(id);
        }
    }

    distance = 0;
    gradient = Eigen::MatrixXd::Zero(tv.rows(), 3);

    for(auto it = map_voxel_points.begin(); it != map_voxel_points.end(); it++)
    {
        int voxel_digit = it->first;
        Eigen::Vector3i voxel_index = digit_to_index(voxel_digit);
        const vector<int>& point_indices = it->second;
        vector<std::pair<int, int>> minimum_distance_between_current_voxel_and_selected_voxels;
        for(int id = 0; id < selected_voxel_indices.size(); id++){
            Eigen::Vector3i selected_voxel_index = selected_voxel_indices[id];
            int selected_voxel_digit = index_to_digit(selected_voxel_index);
            int minimum_distance = computeDistanceVoxelToVoxel(voxel_index, selected_voxel_index);
            minimum_distance_between_current_voxel_and_selected_voxels.push_back({selected_voxel_digit, minimum_distance});
        }

        std::sort(minimum_distance_between_current_voxel_and_selected_voxels.begin(),
                  minimum_distance_between_current_voxel_and_selected_voxels.end(),
                  [](std::pair<int, int> a, std::pair<int, int> b){
            return a.second < b.second;
        });

        for(int id = 0; id < point_indices.size(); id++)
        {
            int point_id = point_indices[id];
            Eigen::Vector3d pt = tv.row(point_id);
            double point_distance = std::numeric_limits<double>::max();
            Eigen::Vector3d point_graident;
            for(int jd = 0; jd < minimum_distance_between_current_voxel_and_selected_voxels.size(); jd++)
            {
                double minimum_distance = minimum_distance_between_current_voxel_and_selected_voxels[jd].second;
                if(point_distance < minimum_distance){
                    break;
                }

                int selected_voxel_digit = minimum_distance_between_current_voxel_and_selected_voxels[jd].first;
                Eigen::Vector3i selected_voxel_index = digit_to_index(selected_voxel_digit);
                double curr_point_voxel_distance;
                Eigen::Vector3d curr_point_voxel_distance_graident;
                computeDiffDistancePointToVoxel(pt,
                                                selected_voxel_index,
                                                curr_point_voxel_distance,
                                                curr_point_voxel_distance_graident);

                std::cout << point_id << ", " << selected_voxel_index.transpose() << ", " << curr_point_voxel_distance << std::endl;

                if(curr_point_voxel_distance < point_distance){
                    point_distance = curr_point_voxel_distance;
                    point_graident = curr_point_voxel_distance_graident;
                }
            }

            distance += point_distance;
            gradient.row(point_id) = point_graident;
        }
    }
}

void MeshVoxelOpt::computeDiffShapeEnegry(const Eigen::MatrixXd &tv,
                                          double &energy,
                                          Eigen::MatrixXd &gradient){
    energy = 0;
    for(int id = 0; id < E_.rows(); id++){
        Eigen::Vector3d pA = tv.row(E_(id, 0));
        Eigen::Vector3d pB = tv.row(E_(id, 1));
        double L = (pA - pB).norm();
        energy += 0.5 * pow(L - l0[id], 2.0);
    }

    gradient = Eigen::MatrixXd::Zero(tv.rows(), 3);
    for(int id = 0; id < E_.rows(); id++)
    {
        int iA = E_(id, 0);
        int iB = E_(id, 1);

        Eigen::Vector3d pA = tv.row(iA);
        Eigen::Vector3d pB = tv.row(iB);
        double L = (pA - pB).norm();

        Eigen::RowVector3d graident_A = (pA - pB) / L;
        Eigen::RowVector3d graident_B = (pB - pA) / L;

        gradient.row(iA) += graident_A * (L - l0[id]);
        gradient.row(iB) += graident_B * (L - l0[id]);
    }

    return;
}