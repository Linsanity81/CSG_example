//
// Created by ziqwang on 08.12.21.
//

#include "MeshVoxelARAP.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "igl/fit_rotations.h"
#include "igl/cotmatrix.h"
#include <iostream>
#include "Eigen/Cholesky"
#include "igl/polar_svd.h"
#include "igl/columnize.h"

void MeshVoxelARAP::readMesh(std::string filename) {
    MeshVoxel::readMesh(filename);
}

void MeshVoxelARAP::precompute_arap_data(const Eigen::VectorXi &b){
    igl::cotmatrix(meshV_, meshF_, L_);
    igl::arap_precomputation(meshV_, meshF_, 3, b, arap_data_);
}

Eigen::MatrixXd MeshVoxelARAP::solve_arap(Eigen::MatrixXd bc, int num_iters){
    Eigen::MatrixXd meshV1 = meshV_;
    arap_data_.max_iter = num_iters;
    igl::arap_solve(bc, arap_data_, meshV1);
    return meshV1;
}

void MeshVoxelARAP::compute_rotation_matrices(const Eigen::MatrixXd &U)
{
    const auto & Udim = U.replicate(arap_data_.dim, 1);
    Eigen::MatrixXd S = arap_data_.CSM * Udim;
    S /= S.array().abs().maxCoeff();
    const int Rdim = arap_data_.dim;
    R_ = Eigen::MatrixXd (Rdim, arap_data_.CSM.rows());
    igl::fit_rotations(S, false, R_);
}

void MeshVoxelARAP::compute_shape_enegry(const Eigen::MatrixXd &meshV1,
                                         double &E,
                                         Eigen::MatrixXd &gradient) const{

    int nV = meshV_.rows();
    Eigen::SparseMatrix<double> Q = -L_;

    Eigen::VectorXd Rcol;
    const int Rdim = arap_data_.dim;
    int num_rots = arap_data_.K.cols()/Rdim/Rdim;
    igl::columnize(R_, num_rots, 2, Rcol);
    Eigen::VectorXd Bcol = -arap_data_.K * Rcol;

    gradient = Eigen::MatrixXd::Zero(nV, 3);
    E = 0;
    for(int id = 0; id < 3; id++){
        Eigen::VectorXd x = meshV1.col(id);
        Eigen::VectorXd b = Bcol.segment(id * nV, nV);
        E += 0.5 * x.dot(Q * x) + x.dot(b);
        gradient.col(id) = Q * x + b;
    }
}

void MeshVoxelARAP::compute_rhs_vectors(Eigen::MatrixXd &rhs) {
    rhs = Eigen::MatrixXd::Zero(meshV_.rows(), 3);

    Eigen::VectorXd Rcol;
    const int Rdim = arap_data_.dim;
    int num_rots = arap_data_.K.cols()/Rdim/Rdim;
    igl::columnize(R_, num_rots,2,Rcol);
    Eigen::VectorXd Bcol = -arap_data_.K * Rcol;

    for(int id = 0; id < 3; id++){
        rhs.col(id) = -Bcol.segment(id * num_rots, num_rots);
    }

    return;
}

void MeshVoxelARAP::compute_energy(const Eigen::MatrixXd &meshV1,
                                   double &E,
                                   Eigen::MatrixXd &gradient) const {
    Eigen::MatrixXd pt_gradientDistance;
    double pt_distance = 0;
    compute_point_to_selected_voxels_distance(meshV1, pt_distance, pt_gradientDistance);

    Eigen::MatrixXd tri_gradientDistance;
    double tri_distance = 0;
    compute_triangle_to_selected_voxels_distance(meshV1, tri_distance, tri_gradientDistance);

    Eigen::MatrixXd gradientShape;
    double shape = 0;
    compute_shape_enegry(meshV1, shape, gradientShape);

    //E = shape * shape_weight_ + pt_distance;
    E = shape * shape_weight_ + pt_distance + tri_distance;
    std::cout << shape << ", " << pt_distance << ", " << tri_distance << std::endl;

    //gradient = gradientShape * shape_weight_ + pt_gradientDistance;
    gradient = gradientShape * shape_weight_ + pt_gradientDistance + tri_gradientDistance;
}

double MeshVoxelARAP::line_search(const Eigen::MatrixXd &x,
                                  const Eigen::MatrixXd &p,
                                  const Eigen::MatrixXd &grad,
                                  double init_step){

    double f_x, f_y;
    Eigen::MatrixXd _;
    double alpha = init_step;
    double beta = 0.5;
    double c = 1e-4;
    compute_energy(x, f_x, _);
    while(true){
        Eigen::MatrixXd y = x + p * alpha;
        compute_energy(y, f_y, _);
        double armijo_condition = f_y - f_x - c * alpha * (p.array() * grad.array()).sum();
        if(armijo_condition <= 0){
            break;
        }
        alpha *= beta;
    }

    return alpha;
}

double MeshVoxelARAP::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const{
    Eigen::MatrixXd tv;
    reshape(x, tv);

    Eigen::MatrixXd gradient;
    double energy;
    compute_energy(tv, energy, gradient);

    flatten(gradient, grad);

    return energy;
}