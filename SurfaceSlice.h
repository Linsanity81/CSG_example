//
// Created by ziqwang on 29.12.21.
//

#ifndef MESHVOXEL_SURFACESLICE_H
#define MESHVOXEL_SURFACESLICE_H

#include <vector>
#include <Eigen/Dense>
#include "fusion.h"
using std::vector;

class SurfaceSlice{
public:

    vector<vector<double>> radius_;

    vector<double> xs_;

    int num_theta_sample_;

    int num_x_sample_;
public:

    SurfaceSlice(double x0, double x1){
        num_theta_sample_ = 128;
        num_x_sample_ = 100;
        for(int id = 1; id <= num_x_sample_; id++){
            xs_.push_back((double) id / (num_x_sample_ + 1) * (x1 - x0) + x0);
        }
    }

    void initSurface(Eigen::Vector3d grids_origin,
                     double grids_width,
                     int grids_size,
                     const vector<Eigen::Vector3i> &voxel_indices);

    void compute_voxel(Eigen::Vector3i index,
                       Eigen::Vector3d grids_origin,
                       double grids_width,
                       Eigen::MatrixXd &V,
                       Eigen::MatrixXi &F);

    void computeMesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F);

    void optimize(double weight, vector<vector<double>> &new_radius);

};


#endif //MESHVOXEL_SURFACESLICE_H
