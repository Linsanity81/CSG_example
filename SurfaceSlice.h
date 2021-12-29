//
// Created by ziqwang on 29.12.21.
//

#ifndef MESHVOXEL_SURFACESLICE_H
#define MESHVOXEL_SURFACESLICE_H

#include <vector>
#include <Eigen/Dense>
using std::vector;

class SurfaceSlice{
public:

    vector<vector<double>> radius_;

    vector<double> xs_;

public:

    void initSurface(Eigen::Vector3d grids_origin,
                     double grids_width,
                     int grids_size,
                     const vector<Eigen::Vector3i> &voxel_indices,
                     const vector<double> &x);

    void compute_voxel(Eigen::Vector3i index,
                       Eigen::Vector3d grids_origin,
                       double grids_width,
                       Eigen::MatrixXd &V,
                       Eigen::MatrixXi &F);

};


#endif //MESHVOXEL_SURFACESLICE_H
