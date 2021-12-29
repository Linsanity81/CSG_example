//
// Created by ziqwang on 28.12.21.
//

#include "SurfaceEvo.h"
#include <iostream>
#include <map>

void SurfaceEvo::compute_ys(const vector<double> &data_yts,
                            const Eigen::VectorXd &xs,
                            Eigen::VectorXd &ys) {
    vector<vector<double>> group_xs;
    divide_xs(xs, group_xs);
    ys = Eigen::VectorXd::Zero(xs.size());
    double index = 0;

    for(int id = 0; id < group_xs.size(); id++)
    {
        if(group_xs.empty()){
            continue;
        }
        else{
            Eigen::Vector4d C;
            C << data_yts[2 * id],
            data_yts[2 * id + 1],
            data_yts[2 * id + 2],
            data_yts[2 * id + 3];

            Eigen::MatrixXd B = computeB(data_xs_[id], data_xs_[id + 1]);
            Eigen::Vector4d A = B.inverse() * C;

            for(int jd = 0; jd < group_xs[id].size(); jd++)
            {
                double x = group_xs[id][jd];
                Eigen::Vector4d X;
                X << x * x * x, x * x, x, 1;
                ys[index] = X.dot(A);
                index ++;
            }
        }
    }
    return;
}

Eigen::MatrixXd SurfaceEvo::computeB(double x0, double x1) {
    Eigen::MatrixXd B(4, 4);
    B << x0 * x0 * x0, x0 * x0, x0, 1,
    3 * x0 * x0, 2 * x0, 1, 0,
    x1 * x1 * x1, x1 * x1, x1, 1,
    3 * x1 * x1, 2 * x1, 1, 0;
    return B;
}

void SurfaceEvo::divide_xs(const Eigen::VectorXd &xs, vector<vector<double>> &group_xs)
{
    int curr = 0;
    group_xs.resize((int) data_xs_.size() - 1);
    for(int id = 0; id < xs.size(); id++){
        double value = xs[id];
        while(value > data_xs_[curr + 1] && curr + 1 < data_xs_.size()){
            curr++;
        }
        group_xs[curr].push_back(value);
    }
}

void SurfaceEvo::computeMesh(const vector<double> &data_yts,
                             Eigen::MatrixXd &V,
                             Eigen::MatrixXi &F){
    int num_x_sample = 100;
    int num_theta_sample = 36;

    Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(num_x_sample, data_xs_.front(), data_xs_.back());
    Eigen::VectorXd ys;
    compute_ys(data_yts, xs, ys);

    V = Eigen::MatrixXd::Zero(num_x_sample * num_theta_sample + 2, 3);

    for(int id = 0; id  < num_x_sample; id++)
    {
        for(int jd = 0; jd < num_theta_sample; jd++)
        {
            double theta = M_PI * 2 / num_theta_sample * jd;
            V(id * num_theta_sample + jd, 0) = xs[id];
            V(id * num_theta_sample + jd, 1) = std::cos(theta) * ys[id];
            V(id * num_theta_sample + jd, 2) = std::sin(theta) * ys[id];
        }
    }
    V(num_x_sample * num_theta_sample, 0) = data_xs_.front();
    V(num_x_sample * num_theta_sample + 1, 0) = data_xs_.back();

    Eigen::MatrixXi LF = Eigen::MatrixXi(2 * (num_x_sample - 1) * num_theta_sample, 3);

    for(int id = 0; id + 1 < num_x_sample; id++)
    {
        for(int jd = 0; jd < num_theta_sample; jd++)
        {
            int iA = id * num_theta_sample + jd;
            int iB = id * num_theta_sample + (jd + 1) % num_theta_sample;
            int iC = iA + num_theta_sample;
            int iD = iB + num_theta_sample;

            LF(id * num_theta_sample * 2 + 2 * jd, 0) = iA;
            LF(id * num_theta_sample * 2 + 2 * jd, 1) = iD;
            LF(id * num_theta_sample * 2 + 2 * jd, 2) = iC;

            LF(id * num_theta_sample * 2 + 2 * jd + 1, 0) = iA;
            LF(id * num_theta_sample * 2 + 2 * jd + 1, 1) = iB;
            LF(id * num_theta_sample * 2 + 2 * jd + 1, 2) = iD;
        }
    }

    Eigen::MatrixXi TF(num_theta_sample, 3);
    for(int jd = 0; jd < num_theta_sample; jd++)
    {
        TF(jd, 0) = jd;
        TF(jd, 2) = (jd + 1) % num_theta_sample;
        TF(jd, 1) = num_x_sample * num_theta_sample;
    }

    Eigen::MatrixXi BF(num_theta_sample, 3);
    for(int jd = 0; jd < num_theta_sample; jd++)
    {
        int index = num_theta_sample * (num_x_sample - 1);
        BF(jd, 0) = jd + index;
        BF(jd, 1) = (jd + 1) % num_theta_sample + index;
        BF(jd, 2) = num_x_sample * num_theta_sample + 1;
    }

    F = Eigen::MatrixXi (TF.rows() + LF.rows() + BF.rows(), 3);
    F.block(0, 0, LF.rows(), 3) = LF;
    F.block(LF.rows(), 0, TF.rows(), 3) = TF;
    F.block(LF.rows() + TF.rows(), 0, BF.rows(), 3) = BF;
//    F << LF, TF;
}

void SurfaceEvo::computeRadius(Eigen::Vector3d grids_origin,
                               double grids_width,
                               int grids_size,
                               const vector<Eigen::Vector3i> &selected_voxel_indices,
                               const vector<double> &xs,
                               vector<double> &radius){
    std::map<int, bool> selected_voxels;

    for(int id = 0; id < selected_voxel_indices.size(); id++){
        Eigen::Vector3i index = selected_voxel_indices[id];
        int digit = index(0) + index(1) * grids_size + index(2) * grids_size * grids_size;
        selected_voxels[digit] = true;
    }

    radius.resize(xs.size(), std::numeric_limits<double>::max());

    for(int digit = 0; digit < grids_size * grids_size * grids_size; digit++)
    {
        if(selected_voxels[digit] == false){
            int ix = digit % grids_size;
            int iy = ((digit - ix) / grids_size) % grids_size;
            int iz = (digit - ix - iy * grids_size) / (grids_size * grids_size);

            double x0 = ix * grids_width + grids_origin(0);
            double x1 = x0 + grids_width;

            double y0 = iy * grids_width + grids_origin(1);
            double y1 = y0 + grids_width;

            double z0 = iz * grids_width + grids_origin(2);
            double z1 = z0 + grids_width;

            double r0 = std::sqrt(y0 * y0 + z0 * z0);
            double r1 = std::sqrt(y1 * y1 + z0 * z0);
            double r2 = std::sqrt(y0 * y0 + z1 * z1);
            double r3 = std::sqrt(y1 * y1 + z1 * z1);

            double r = std::min(std::min(r0, r1), std::min(r2, r3));
//            std::cout << ix << ", " << iy << ", " << iz << ":\t";
            for(int id = 0; id < xs.size(); id++)
            {
                if(xs[id] >= x0 && xs[id] <= x1)
                {
//                    std::cout << id << " ";
                    radius[id] = std::min(radius[id], r);
                }

                if(xs[id] > x1){
                    break;
                }
            }
//            std::cout << std::endl;
        }
    }
}


void SurfaceEvo::compute_constraints(const vector<double> &xs,
                         vector<double> &radius,
                         Eigen::MatrixXd &Mat,
                         Eigen::VectorXd &b){

    Mat = Eigen::MatrixXd::Zero(xs.size(), 2 * data_xs_.size());
    b = Eigen::VectorXd::Zero(xs.size());

    for(int id = 0; id < xs.size(); id++)
    {
        double x = xs[id];
        double x0 = 0, x1 = 0;
        int interval_index = 0;
        for(int jd = 0; jd + 1 < data_xs_.size(); jd++)
        {
            x0 = data_xs_[jd];
            x1 = data_xs_[jd + 1];
            if(x0 <= x && x < x1){
                interval_index = jd;
                break;
            }
        }

        Eigen::MatrixXd matB = computeB(x0, x1);
        Eigen::VectorXd X(4);
        X(0) = x * x * x;
        X(1) = x * x;
        X(2) = x;
        X(3) = 1;

        Eigen::RowVector4d coeff = X.transpose() * matB.inverse();
        Mat.block(id, interval_index * 2, 1, 4) = coeff;
        b(id) = radius[id];
    }

}