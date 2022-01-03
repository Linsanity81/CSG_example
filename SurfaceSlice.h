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
        num_theta_sample_ = 32;
        num_x_sample_ = 100;
        for(int id = 1; id <= num_x_sample_; id++){
            xs_.push_back((double) id / (num_x_sample_ + 1) * (x1 - x0) + x0);
        }
    }

    Eigen::Vector3i digit_to_index(int digit, Eigen::Vector3i grids_size) const{
        int ix = digit % grids_size[0];
        int iy = ((digit - ix) / grids_size[0]) % grids_size[1];
        int iz = (digit - ix - iy * grids_size[0]) / (grids_size[0] * grids_size[1]);
        return Eigen::Vector3i(ix, iy, iz);
    }

    int index_to_digit(Eigen::Vector3i index, Eigen::Vector3i grids_size) const
    {
        int ix = index[0];
        int iy = index[1];
        int iz = index[2];
        if(ix >= 0 && ix < grids_size[0] && iy >= 0 && iy < grids_size[1] && iz >= 0 && iz < grids_size[2]){
            return ix + iy * grids_size[0] + iz * grids_size[0] * grids_size[1];
        }
        else{
            return -1;
        }

    }

    void initSurface(Eigen::Vector3d grids_origin,
                     double grids_width,
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
