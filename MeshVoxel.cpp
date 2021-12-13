//
// Created by ziqwang on 15.11.21.
//

#include "MeshVoxel.h"
#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
void MeshVoxel::readMesh(std::string filename) {
    igl::readOBJ(filename, meshV_, meshF_);
}

void MeshVoxel::voxelization(vector<Eigen::MatrixXd> &Vs,
                             vector<Eigen::MatrixXi> &Fs,
                             vector<double> &volumes,
                             vector<vector<double>> &areas,
                             vector<Eigen::Vector3i> &voxel_indices)
{
    int num_of_voxels = grids_size_ * grids_size_ * grids_size_;

    Vs.resize(num_of_voxels);
    Fs.resize(num_of_voxels);

    volumes.resize(num_of_voxels);
    areas.resize(num_of_voxels);
    voxel_indices.resize(num_of_voxels);

    tbb::parallel_for( tbb::blocked_range<int>(0, num_of_voxels),
                       [&](tbb::blocked_range<int> r) {
                           for (int id = r.begin(); id < r.end(); ++id)
                           {
                               Eigen::MatrixXd V;
                               Eigen::MatrixXi F;
                               Eigen::Vector3i index = digit_to_index(id);
                               volumes[id] = compute_intersec(index, V, F);
                               Vs[id] = V;
                               Fs[id] = F;
                               areas[id] = compute_contacts(index, V, F);
                               voxel_indices[id] = index;
                           }
                       });

    double min_volume = 1E-6;

    vector<Eigen::MatrixXd> Vs_tmp;
    vector<Eigen::MatrixXi> Fs_tmp;
    vector<double> volumes_tmp;
    vector<vector<double>> areas_tmp;
    vector<Eigen::Vector3i> voxel_indices_tmp;

    for(int id = 0; id < Vs.size(); id++){
        if(volumes[id] > min_volume){
            Vs_tmp.push_back(Vs[id]);
            Fs_tmp.push_back(Fs[id]);
            volumes_tmp.push_back(volumes[id]);
            areas_tmp.push_back(areas[id]);
            voxel_indices_tmp.push_back(voxel_indices[id]);
        }
    }
    Vs = Vs_tmp;
    Fs = Fs_tmp;
    volumes = volumes_tmp;
    areas = areas_tmp;
    voxel_indices = voxel_indices_tmp;

    return;
}

std::vector<double> MeshVoxel::compute_contacts(Eigen::Vector3i index,
                                     const Eigen::MatrixXd &V,
                                     const Eigen::MatrixXi &F){
    std::vector<double> area;
    area.resize(6, 0);
    double eps = 1E-5;
    for(int id = 0; id < 6; id++)
    {
        Eigen::Vector3i curr_index = index + Eigen::Vector3i((dX[id] + 1) / 2,
                                                             (dY[id] + 1) / 2,
                                                             (dZ[id] + 1) / 2);
        Eigen::Vector3d offset = curr_index.cast<double>();
        offset *= grids_width_;
        Eigen::Vector3d origin = grids_origin_ + offset;
        Eigen::Vector3d normal(dX[id], dY[id], dZ[id]);
        for(int jd = 0; jd < F.rows(); jd++){
            Eigen::Vector3d v0 = V.row(F(jd, 0));
            Eigen::Vector3d v1 = V.row(F(jd, 1));
            Eigen::Vector3d v2 = V.row(F(jd, 2));
            Eigen::Vector3d fn = (v1 - v0).cross(v2 - v0);
            if(fn.norm() > eps)
            {
                double angle = fn.normalized().dot(normal);
                double distance = std::abs((v0 - origin).dot(normal));
                if(angle > 1 - eps && distance < eps){
                    area[id] += fn.norm() * 0.5;
                }
            }
        }
    }

    return area;
}

double MeshVoxel::compute_intersec(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    Eigen::MatrixXd voxelV;
    Eigen::MatrixXi voxelF;

    compute_voxel(index, voxelV, voxelF);

    Eigen::VectorXi J;
    igl::copyleft::cgal::mesh_boolean(meshV_, meshF_, voxelV, voxelF, igl::MeshBooleanType::MESH_BOOLEAN_TYPE_INTERSECT, V, F, J);

    if(F.rows() != 0){
        Eigen::MatrixXd V2(V.rows() + 1, V.cols());
        V2.topRows(V.rows()) = V;
        V2.bottomRows(1).setZero();
        Eigen::MatrixXi T(F.rows(), 4);
        T.leftCols(3) = F;
        T.rightCols(1).setConstant(V.rows());
        Eigen::VectorXd vol;
        igl::volume(V2, T, vol);
        return std::abs(vol.sum());
    }
    else{
        return -1;
    }
}

void MeshVoxel::compute_voxel(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    V = Eigen::MatrixXd::Zero(8, 3);
    F = Eigen::MatrixXi::Zero(12, 3);

    V << 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 1.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 1.0,
      1.0, 1.0, 0.0,
      1.0, 1.0, 1.0;

    F << 0, 6, 4,
    0, 2, 6,
    0, 3, 2,
    0, 1, 3,
    2, 7, 6,
    2, 3, 7,
    4, 6, 7,
    4, 7, 5,
    0, 4, 5,
    0, 5, 1,
    1, 5, 7,
    1, 7, 3;

    for(int id = 0; id < V.rows(); id++){
        V.row(id) *= grids_width_;
        V.row(id) += grids_origin_.transpose();
        Eigen::Vector3d offset = index.cast<double>();
        offset *= grids_width_;
        V.row(id) += offset.transpose();
    }

    return;
}

double MeshVoxel::computeDistanceVoxelToVoxel(Eigen::Vector3i voxelA, Eigen::Vector3i voxelB) const{
    Eigen::Vector3i distance = voxelA - voxelB;
    double minimum_distance = 0;
    for(int kd = 0; kd < 3; kd++){
        double d = std::abs(distance[kd]);
        minimum_distance += d > 0 ? pow((d - 1) * grids_width_, 2): 0;
    }
    return minimum_distance;
}

void MeshVoxel::computeDiffDistancePointToVoxel(Eigen::Vector3d pt,
                                                   Eigen::Vector3i voxel_index,
                                                   double &distance,
                                                   Eigen::Vector3d &gradient) const{
    distance = 0;
    gradient = Eigen::Vector3d(0, 0, 0);
    for(int kd = 0; kd < 3; kd++)
    {
        double max_voxel_coord = grids_origin_[kd] + (voxel_index[kd] + 1) * grids_width_;
        double min_voxel_coord = grids_origin_[kd] + voxel_index[kd] * grids_width_;

        if(pt[kd] > max_voxel_coord)
        {
//            distance += pow(pt[kd] - max_voxel_coord, 2.0);
//            gradient[kd] = 2 * (pt[kd] - max_voxel_coord);
            distance += pt[kd] - max_voxel_coord;
            gradient[kd] = 1;
        }
        else if(pt[kd] < min_voxel_coord){
//            distance += pow(min_voxel_coord - pt[kd], 2.0);
//            gradient[kd] = 2 * (pt[kd] - min_voxel_coord);
            distance += min_voxel_coord - pt[kd];
            gradient[kd] = -1;
        }
    }
}

void MeshVoxel::computeSelectedVoxels(vector<double> &volumes, vector<Eigen::Vector3i> &voxel_indices)
{
    selected_voxel_indices_.clear();
    for(int id = 0; id < volumes.size(); id++){
        if(volumes[id] > minimum_volume_){
            selected_voxel_indices_.push_back(voxel_indices[id]);
        }
    }

    return;
}


void MeshVoxel::computeDiffDistanceToSelectedVoxels(const Eigen::MatrixXd &tv,
                                                       double &distance,
                                                       Eigen::MatrixXd &gradient) const{
    distance = 0;
    gradient = Eigen::MatrixXd::Zero(tv.rows(), 3);
    for(int id = 0; id < tv.rows(); id++)
    {
        int point_id = id;
        Eigen::Vector3d pt = tv.row(point_id);
        double point_distance = std::numeric_limits<double>::max();
        Eigen::Vector3d point_gradient;
        for(int jd = 0; jd < selected_voxel_indices_.size(); jd++)
        {

            Eigen::Vector3i selected_voxel_index = selected_voxel_indices_[jd];
            double curr_point_voxel_distance;
            Eigen::Vector3d curr_point_voxel_distance_graident;
            computeDiffDistancePointToVoxel(pt,
                                            selected_voxel_index,
                                            curr_point_voxel_distance,
                                            curr_point_voxel_distance_graident);

            if(curr_point_voxel_distance < point_distance){
                point_distance = curr_point_voxel_distance;
                point_gradient = curr_point_voxel_distance_graident;
            }
        }

        distance += point_distance;
        gradient.row(point_id) = point_gradient;
    }
}
