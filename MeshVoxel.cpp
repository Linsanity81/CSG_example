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
    std::vector<double> volumes(num_of_voxels);
    std::vector<std::vector<double>> areas(num_of_voxels);
    tbb::parallel_for( tbb::blocked_range<int>(0, num_of_voxels),
                       [&](tbb::blocked_range<int> r) {
                           for (int id = r.begin(); id < r.end(); ++id)
                           {
                               Eigen::MatrixXd V;
                               Eigen::MatrixXi F;
                               Eigen::Vector3i index = to_xyz(id, M);
                               volumes[id] = compute_intersec(index, V, F);
                               areas[id] = compute_contacts(index, V, F);
                           }
                       });

    double min_volume = 1E-6;
    double min_contact = grids_width_ * grids_width_ * 0.3;

    for (int id = 0; id < num_of_voxels; ++id) {
        Eigen::Vector3i index = to_xyz(id, M);
        std::cout << id << "\t" << index.transpose() << "\t" << volumes[id] << std::endl;
    }

    for(int id = 0; id < num_of_voxels; id++)
    {
        Eigen::Vector3i index = to_xyz(id, M);
        for(int jd = 0; jd < 6; jd++)
        {
            Eigen::Vector3i nindex = index + Eigen::Vector3i(dX[jd], dY[jd], dZ[jd]);
            int nid = to_index(nindex, M);
            if(nid > id)
            {
                std::cout << id << "\t" << nid << "\t" << areas[id][jd] << std::endl;
            }
        }
    }
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
