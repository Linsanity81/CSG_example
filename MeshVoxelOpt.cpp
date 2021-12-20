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

void MeshVoxelOpt::computeDiffShapeEnergy(const Eigen::MatrixXd &tv,
                                          double &energy,
                                          Eigen::MatrixXd &gradient) const{
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

double MeshVoxelOpt::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &grad) const
{

    Eigen::MatrixXd tv;
    reshape(x, tv);

    double distance_energy, shape_energy;
    Eigen::MatrixXd distance_gradient, shape_gradient;
    compute_point_to_selected_voxels_distance(tv, distance_energy, distance_gradient);
    computeDiffShapeEnergy(tv, shape_energy, shape_gradient);
    Eigen::MatrixXd gradient = distance_gradient + weight_shape_energy * shape_gradient;

    flatten(gradient, grad);
    return distance_energy + weight_shape_energy * shape_energy;
}
