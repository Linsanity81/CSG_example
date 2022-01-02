//
// Created by ziqwang on 29.12.21.
//

#include "SurfaceSlice.h"
#include <map>
#include "igl/ray_mesh_intersect.h"
#include "igl/ray_box_intersect.h"
#include <iostream>

void SurfaceSlice::initSurface(Eigen::Vector3d grids_origin,
                               double grids_width,
                               int grids_size,
                               const vector<Eigen::Vector3i> &voxel_indices)
                               {
    std::map<int, bool> selected_voxels;

    for(int id = 0; id < voxel_indices.size(); id++){
        Eigen::Vector3i index = voxel_indices[id];
        int digit = index(0) + index(1) * grids_size + index(2) * grids_size * grids_size;
        selected_voxels[digit] = true;
    }

    radius_.resize(xs_.size());
    for(int id = 0; id < xs_.size(); id++){
        radius_[id].resize(num_theta_sample_, std::numeric_limits<double>::max());
    }
    
    for(int digit = 0; digit < grids_size * grids_size * grids_size; digit++)
    {
        if(selected_voxels[digit] == false){
            int ix = digit % grids_size;
            int iy = ((digit - ix) / grids_size) % grids_size;
            int iz = (digit - ix - iy * grids_size) / (grids_size * grids_size);
            Eigen::Vector3i index(ix, iy, iz);
            double x0 = ix * grids_width + grids_origin(0);
            double x1 = x0 + grids_width;

            double y0 = iy * grids_width + grids_origin(1);
            double y1 = y0 + grids_width;

            double z0 = iz * grids_width + grids_origin(2);
            double z1 = z0 + grids_width;

            Eigen::Vector3d min_box(x0, y0, z0);
            Eigen::Vector3d max_box(x1, y1, z1);
            Eigen::AlignedBox<double, 3> box(min_box, max_box);


//            Eigen::MatrixXd V;
//            Eigen::MatrixXi F;
//            compute_voxel(index, grids_origin, grids_width, V,F);

            Eigen::Matrix<double, 1, 3, 1, 1, 3> pt((x0 + x1) / 2.0, 0, 0);
            vector<double> voxel_radius;

            for(int id = 0; id < num_theta_sample_; id++){
                double angle = 2.0 * M_PI / num_theta_sample_ * id;
                Eigen::Matrix<double, 1, 3, 1, 1, 3> drt(0, cos(angle), sin(angle));

                const double t0 = 0.0;
                const double t1 = 10.0;
                double tmin, tmax;
                if(igl::ray_box_intersect(pt, drt, box, t0, t1, tmin, tmax)){
                    voxel_radius.push_back(tmin);
                }
                else{
                    voxel_radius.push_back(std::numeric_limits<double>::max());
                }
            }


            for(int id = 0; id < xs_.size(); id++)
            {
                if(xs_[id] >= x0 && xs_[id] <= x1)
                {
                    for(int jd = 0; jd < num_theta_sample_; jd++){
                        radius_[id][jd] = std::min(radius_[id][jd], voxel_radius[jd]);
                    }
                }

                if(xs_[id] > x1){
                    break;
                }
            }
        }
    }
}

void SurfaceSlice::compute_voxel(Eigen::Vector3i index,
                                 Eigen::Vector3d grids_origin,
                                 double grids_width,
                                 Eigen::MatrixXd &V,
                                 Eigen::MatrixXi &F) {
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
        V.row(id) *= grids_width;
        V.row(id) += grids_origin.transpose();
        Eigen::Vector3d offset = index.cast<double>();
        offset *= grids_width;
        V.row(id) += offset.transpose();
    }

    return;
}

void SurfaceSlice::computeMesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F){

    V = Eigen::MatrixXd::Zero(num_x_sample_ * num_theta_sample_ + 2, 3);

    for(int id = 0; id  < num_x_sample_; id++)
    {
        for(int jd = 0; jd < num_theta_sample_; jd++)
        {
            double theta = M_PI * 2 / num_theta_sample_ * jd;
            V(id * num_theta_sample_ + jd, 0) = xs_[id];
            V(id * num_theta_sample_ + jd, 1) = std::cos(theta) * radius_[id][jd];
            V(id * num_theta_sample_ + jd, 2) = std::sin(theta) * radius_[id][jd];
        }
    }
    V(num_x_sample_ * num_theta_sample_, 0) = xs_.front();
    V(num_x_sample_ * num_theta_sample_ + 1, 0) = xs_.back();

    Eigen::MatrixXi LF = Eigen::MatrixXi(2 * (num_x_sample_ - 1) * num_theta_sample_, 3);

    for(int id = 0; id + 1 < num_x_sample_; id++)
    {
        for(int jd = 0; jd < num_theta_sample_; jd++)
        {
            int iA = id * num_theta_sample_ + jd;
            int iB = id * num_theta_sample_ + (jd + 1) % num_theta_sample_;
            int iC = iA + num_theta_sample_;
            int iD = iB + num_theta_sample_;

            LF(id * num_theta_sample_ * 2 + 2 * jd, 0) = iA;
            LF(id * num_theta_sample_ * 2 + 2 * jd, 1) = iD;
            LF(id * num_theta_sample_ * 2 + 2 * jd, 2) = iC;

            LF(id * num_theta_sample_ * 2 + 2 * jd + 1, 0) = iA;
            LF(id * num_theta_sample_ * 2 + 2 * jd + 1, 1) = iB;
            LF(id * num_theta_sample_ * 2 + 2 * jd + 1, 2) = iD;
        }
    }

    Eigen::MatrixXi TF(num_theta_sample_, 3);
    for(int jd = 0; jd < num_theta_sample_; jd++)
    {
        TF(jd, 0) = jd;
        TF(jd, 2) = (jd + 1) % num_theta_sample_;
        TF(jd, 1) = num_x_sample_ * num_theta_sample_;
    }

    Eigen::MatrixXi BF(num_theta_sample_, 3);
    for(int jd = 0; jd < num_theta_sample_; jd++)
    {
        int index = num_theta_sample_ * (num_x_sample_ - 1);
        BF(jd, 0) = jd + index;
        BF(jd, 1) = (jd + 1) % num_theta_sample_ + index;
        BF(jd, 2) = num_x_sample_ * num_theta_sample_ + 1;
    }

    F = Eigen::MatrixXi (TF.rows() + LF.rows() + BF.rows(), 3);
    F.block(0, 0, LF.rows(), 3) = LF;
    F.block(LF.rows(), 0, TF.rows(), 3) = TF;
    F.block(LF.rows() + TF.rows(), 0, BF.rows(), 3) = BF;
    //    F << LF, TF;
}