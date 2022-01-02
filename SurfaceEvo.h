//
// Created by ziqwang on 28.12.21.
//

#ifndef MESHVOXEL_SURFACEEVO_H
#define MESHVOXEL_SURFACEEVO_H

#include <vector>
#include <Eigen/Dense>
using std::vector;

class SurfaceEvo {

public:

    vector<double> data_xs_;

public:

    SurfaceEvo(const vector<double> &data_xs){
        data_xs_ = data_xs;
    }

    void compute_ys(const vector<double> &data_yts,
                    const Eigen::VectorXd &xs,
                    Eigen::VectorXd &ys);

    Eigen::MatrixXd computeB(double x0, double x1);

    void divide_xs(const Eigen::VectorXd &xs,
                   vector<vector<double>> &group_xs);


    void computeMesh(const vector<double> &data_yts, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

    void computeRadius(Eigen::Vector3d grids_origin,
                       double grids_width,
                       int grids_size,
                       const vector<Eigen::Vector3i> &selected_voxel_indices,
                       const vector<double> &xs,
                       vector<double> &radius);

    void compute_constraints(const vector<double> &xs,
                             vector<double> &radius,
                             Eigen::MatrixXd &Mat,
                             Eigen::VectorXd &b);


};


#endif //MESHVOXEL_SURFACEEVO_H
