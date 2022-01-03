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
                               const vector<Eigen::Vector3i> &voxel_indices)
                               {
    std::map<int, bool> selected_voxels;

    int max_index[3] ={0, 0, 0};
    int min_index[3] = {std::numeric_limits<int>::max(),
                        std::numeric_limits<int>::max(),
                        std::numeric_limits<int>::max()};

    for(int id = 0; id < voxel_indices.size(); id++)
    {
        Eigen::Vector3i index = voxel_indices[id];
        for(int jd = 0; jd < 3; jd++){
            max_index[jd] = std::max(max_index[jd], index[jd]);
            min_index[jd] = std::min(min_index[jd], index[jd]);
        }
    }

    //both end add 1 voxels
    Eigen::Vector3i grids_size(max_index[0] - min_index[0] + 3,
                               max_index[1] - min_index[1] + 3,
                               max_index[2] - min_index[2] + 3);

    Eigen::Vector3i delta_index(min_index[0] - 1,
                                min_index[1] - 1,
                                min_index[2] - 1);

    for(int id = 0; id < voxel_indices.size(); id++){
        Eigen::Vector3i index = voxel_indices[id];
        int digit = index_to_digit(index - delta_index, grids_size);
        selected_voxels[digit] = true;
    }

    radius_.resize(xs_.size());
    for(int id = 0; id < xs_.size(); id++){
        radius_[id].resize(num_theta_sample_, std::numeric_limits<double>::max());
    }
    
    for(int digit = 0; digit < grids_size[0] * grids_size[1] * grids_size[2]; digit++)
    {
        if(selected_voxels[digit] == false)
        {
            Eigen::Vector3i index = digit_to_index(digit, grids_size);
            index = index + delta_index;

            int ix = index(0);
            int iy = index(1);
            int iz = index(2);

            double x0 = ix * grids_width + grids_origin(0);
            double x1 = x0 + grids_width;

            double y0 = iy * grids_width + grids_origin(1);
            double y1 = y0 + grids_width;

            double z0 = iz * grids_width + grids_origin(2);
            double z1 = z0 + grids_width;

            Eigen::Vector3d min_box(x0, y0, z0);
            Eigen::Vector3d max_box(x1, y1, z1);
            Eigen::AlignedBox<double, 3> box(min_box, max_box);

            Eigen::Matrix<double, 1, 3, 1, 1, 3> pt((x0 + x1) / 2.0, 0, 0);
            vector<double> voxel_radius;

            for(int id = 0; id < num_theta_sample_; id++)
            {
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

void SurfaceSlice::optimize(double weight, vector<vector<double>> &new_radius){
    int NX = num_x_sample_;
    int NT = num_theta_sample_;
    int nvar = NX * NT;

    vector<double> radius_upper_bound;
    for(int id = 0; id < radius_.size(); id++){
        for(int jd = 0; jd < radius_[id].size(); jd++){
            radius_upper_bound.push_back(radius_[id][jd]);
        }
    }

    auto mosek_radius_upper_bound = monty::new_array_ptr<double>(radius_upper_bound);

    mosek::fusion::Model::t M = new mosek::fusion::Model("quadratic");
    auto _M = monty::finally([&]() { M->dispose(); });

    mosek::fusion::Variable::t var_r  = M->variable("radius", nvar, mosek::fusion::Domain::inRange(0.0, mosek_radius_upper_bound));
    mosek::fusion::Variable::t var_g = M->variable("gap", nvar);
    M->constraint("con_gap", mosek::fusion::Expr::add(var_r, var_g), mosek::fusion::Domain::equalsTo(mosek_radius_upper_bound));

    mosek::fusion::Variable::t var_sh = M->variable("smooth_horizontal", nvar);
    mosek::fusion::Variable::t var_sv = M->variable("smooth_vertical", (NX - 2) * NT);

    int icon = 0;
    for(int id = 0; id < NX; id++)
    {
        for(int jd = 0; jd < NT; jd++)
        {
            int prev_iv = id * NT + (jd - 1 + NT) % NT;
            int next_iv = id * NT + (jd + 1) % NT;
            int iv = id * NT + jd;
            auto expression = mosek::fusion::Expr::add(var_r->index(prev_iv), var_r->index(next_iv));
            expression = mosek::fusion::Expr::sub(expression, mosek::fusion::Expr::mul(2.0, var_r->index(iv)));
            expression = mosek::fusion::Expr::mul(expression, sqrt(weight));
            expression = mosek::fusion::Expr::sub(expression, var_sh->index(iv));
            std::string con_str = "contraints_hortizontal_" + std::to_string(icon);
            M->constraint(con_str, expression, mosek::fusion::Domain::equalsTo(0.0));
            icon++;
        }
    }

    icon = 0;
    for(int id = 1; id + 1 < NX; id++)
    {
        for(int jd = 0; jd < NT; jd++)
        {
            int prev_iv = (id - 1) * NT + jd;
            int next_iv = (id + 1) * NT + jd;
            int iv = id * NT + jd;

            auto expression = mosek::fusion::Expr::add(var_r->index(prev_iv), var_r->index(next_iv));
            expression = mosek::fusion::Expr::sub(expression, mosek::fusion::Expr::mul(2.0, var_r->index(iv)));
            expression = mosek::fusion::Expr::mul(expression, sqrt(weight));
            expression = mosek::fusion::Expr::sub(expression, var_sv->index(icon));
            std::string con_str = "contraints_vertical_" + std::to_string(icon);
            M->constraint(con_str, expression, mosek::fusion::Domain::equalsTo(0.0));
            icon++;
        }
    }

    mosek::fusion::Variable::t var_obj = M->variable(1);
    std::shared_ptr<monty::ndarray<mosek::fusion::Variable::t,1>> varlist
                                                    = monty::new_array_ptr<mosek::fusion::Variable::t,1>({var_obj, var_g, var_sh, var_sv});

    mosek::fusion::Variable::t stack_var = mosek::fusion::Var::vstack(varlist);

    M->constraint("obj_contrain", stack_var, mosek::fusion::Domain::inQCone());

    M->objective("obj", mosek::fusion::ObjectiveSense::Minimize, var_obj);

    M->solve();

    monty::ndarray<double, 1> var_radius   = *(var_r->level());
    new_radius.clear();
    for(int id = 0; id < NX; id++)
    {
        new_radius.push_back(vector<double>());
        for(int jd = 0; jd < NT; jd++){
            int iv = id * NT + jd;
            new_radius[id].push_back(var_radius[iv]);
        }
    }

    return;
}