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

void MeshVoxel::voxelization(int M)
{
    int num_of_voxels = M * M * M;
    std::vector<double> volume(num_of_voxels);
    tbb::parallel_for( tbb::blocked_range<int>(0, num_of_voxels),
                       [&](tbb::blocked_range<int> r) {
                           for (int id = r.begin(); id < r.end(); ++id)
                           {
                               Eigen::MatrixXd V;
                               Eigen::MatrixXi F;
                               int ix = id % 5;
                               int iy = ((id - ix) / 5) % 5;
                               int iz = (id - ix - iy * 5) / 25;
                               volume[id] = compute_intersec(Eigen::Vector3i(ix, iy, iz), V, F);
                           }
                       });

    for (int id = 0; id < num_of_voxels; ++id) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        int ix = id % 5;
        int iy = ((id - ix) / 5) % 5;
        int iz = (id - ix - iy * 5) / 25;
        std::cout << ix << " " << iy << " " << iz << ": " << volume[id] << std::endl;
    }
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
