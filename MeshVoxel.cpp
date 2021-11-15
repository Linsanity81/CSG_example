//
// Created by ziqwang on 15.11.21.
//

#include "MeshVoxel.h"

void MeshVoxel::readMesh(std::string filename) {
    igl::readOBJ(filename, meshV_, meshF_);
}

double MeshVoxel::intersec(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    Eigen::MatrixXd voxelV;
    Eigen::MatrixXi voxelF;

    voxel(index, voxelV, voxelF);

    Eigen::VectorXi J;
    igl::copyleft::cgal::mesh_boolean(meshV_, meshF_, voxelV, voxelF, igl::MeshBooleanType::MESH_BOOLEAN_TYPE_INTERSECT, V, F, J);

//    if(F.rows() != 0){
//        Eigen::VectorXd vol;
//        igl::volume(V, F, vol);
//        return vol.array().abs().sum();
//    }
//    else{
        return -1;
//    }
}

void MeshVoxel::voxel(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
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
    std::cout << V << std::endl;

    return;
}
