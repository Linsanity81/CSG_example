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


void MeshVoxelARAP::readMesh(std::string filename) {
    MeshVoxel::readMesh(filename);
    compute_cotmatrix();
}

void MeshVoxelARAP::compute_cotmatrix() {
    Eigen::SparseMatrix<double> Cot;
    igl::cotmatrix(meshV_, meshF_, Cot);

    L_ = Eigen::SparseMatrix<double, Eigen::RowMajor>(Cot.rows(), Cot.cols());
    L_ = -Cot;
}

void MeshVoxelARAP::compute_rotation_matrix(Eigen::MatrixXd P0, Eigen::MatrixXd P1, Eigen::MatrixXd D, Eigen::MatrixXd &R)
{
    Eigen::MatrixXd S = P0 * D * P1.transpose();
    igl::fit_rotations(S, false, R);
}

void MeshVoxelARAP::compute_rotation_matrices(const Eigen::MatrixXd &meshV1, vector<Eigen::MatrixXd> &Rs)
{
    Rs.clear();
    for(int id = 0; id < L_.rows(); id++)
    {
        vector<int> neighbour_vertices;
        vector<double> neighbour_coefficents;
        for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L_, id); it; ++it)
        {
            if(it.col() != id){
                neighbour_vertices.push_back(it.col());
                neighbour_coefficents.push_back(it.value());
            }
        }

        Eigen::Vector3d X0, x0;
        X0 = meshV_.row(id);
        x0 = meshV1.row(id);

        Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(3, neighbour_vertices.size());
        Eigen::MatrixXd P1 = Eigen::MatrixXd::Zero(3, neighbour_vertices.size());
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(neighbour_vertices.size(), neighbour_vertices.size());
        for(int jd = 0; jd < neighbour_vertices.size(); jd++)
        {
            Eigen::Vector3d X1 = meshV_.row(neighbour_vertices[jd]);
            Eigen::Vector3d x1 = meshV1.row(neighbour_vertices[jd]);
            P0.col(jd) = X0 - X1;
            P1.col(jd) = x0 - x1;
            D(jd, jd) = -neighbour_coefficents[jd];
        }
        Eigen::MatrixXd R;
        compute_rotation_matrix(P0, P1, D, R);
        Rs.push_back(R);
    }
}

void MeshVoxelARAP::compute_shape_enegry(const Eigen::MatrixXd &meshV1,
                          const vector<Eigen::MatrixXd> &Rs,
                          double &E,
                          Eigen::MatrixXd &gradient){

    gradient = Eigen::MatrixXd::Zero(meshV_.rows(), 3);

    E = 0;
    for(int iv = 0; iv < meshV_.rows(); iv++)
    {
        Eigen::Vector3d derivative(0, 0, 0);
        for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L_, iv); it; ++it)
        {
            if(it.col() != iv){
                int jv = it.col();
                Eigen::Vector3d pi_prime = meshV1.row(iv);
                Eigen::Vector3d pj_prime = meshV1.row(jv);
                Eigen::Vector3d pi = meshV_.row(iv);
                Eigen::Vector3d pj = meshV_.row(jv);
                derivative += 4 * abs(-it.value()) * ((pi_prime - pj_prime) - 0.5 * (Rs[iv] + Rs[jv]) * (pi - pj));

                if(((pi_prime - pj_prime) - Rs[iv] * (pi - pj)).squaredNorm() > 1E-5){
                    std::cout << Rs[iv] << std::endl;
                    std::cout << (pi - pj).transpose() << std::endl;
                    std::cout << (pi_prime - pj_prime).transpose() << std::endl;
                }

                E += ((pi_prime - pj_prime) - Rs[iv] * (pi - pj)).squaredNorm() * abs(-it.value());
            }
        }
        gradient.row(iv) = derivative;
    }
}

void MeshVoxelARAP::compute_rhs_vectors(const vector<Eigen::MatrixXd> &Rs, Eigen::MatrixXd &rhs) {
    rhs = Eigen::MatrixXd::Zero(meshV_.rows(), 3);
    for(int iv = 0; iv < meshV_.rows(); iv++)
    {
        Eigen::Vector3d rhs_i(0, 0, 0);
        for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L_, iv); it; ++it)
        {
            if(it.col() != iv){
                int jv = it.col();
                Eigen::Vector3d pi = meshV_.row(iv);
                Eigen::Vector3d pj = meshV_.row(jv);
                Eigen::MatrixXd Ri = Rs[iv];
                Eigen::MatrixXd Rj = Rs[jv];
                rhs_i += -it.value() / 2 * (Ri + Rj) * (pi - pj);
            }
        }
        rhs.row(iv) = rhs_i;
    }

    return;
}

Eigen::MatrixXd MeshVoxelARAP::deform(Eigen::VectorXi b, Eigen::MatrixXd bc, int num_iters) {

    Eigen::MatrixXd meshV1 = meshV_;
    vector<Eigen::MatrixXd> Rs;
    Eigen::MatrixXd rhs;

    int nV = meshV_.rows();
    std::map<int, int> map_nonfixed_vertices_to_new_indices;
    std::map<int, bool> map_fixed_vertices_as_true;
    std::map<int, Eigen::Vector3d> map_fixed_vertices_values;
    for(int id = 0; id < b.rows(); id++){
        int iv = b[id];
        map_fixed_vertices_as_true[iv] = true;
        map_fixed_vertices_values[iv] = bc.row(id);
    }

    int count = 0;
    for(int id = 0; id < nV; id++)
    {
        if(map_fixed_vertices_as_true.find(id) == map_fixed_vertices_as_true.end()){
            map_nonfixed_vertices_to_new_indices[id] = count;
            count++;
        }
    }

    Eigen::SparseMatrix<double> L_star(nV - b.rows(), nV - b.rows());
    vector<Eigen::Triplet<double>> triplist;
    for(int iv = 0; iv < meshV_.rows(); iv++)
    {
        if(map_fixed_vertices_as_true.find(iv) == map_fixed_vertices_as_true.end())
        {
            for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L_, iv); it; ++it)
            {
                int jv = it.col();
                if(map_fixed_vertices_as_true.find(jv) == map_fixed_vertices_as_true.end()){
                    double value = it.value();
                    int new_iv = map_nonfixed_vertices_to_new_indices[iv];
                    int new_jv = map_nonfixed_vertices_to_new_indices[jv];
                    triplist.push_back(Eigen::Triplet<double>(new_iv, new_jv, value));
                }
            }
        }
    }

    L_star.setFromTriplets(triplist.begin(), triplist.end());
    L_star.finalize();
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> lu;
    lu.compute(L_star);

    while(num_iters --){
        compute_rotation_matrices(meshV1, Rs);
        compute_rhs_vectors(Rs, rhs);

        Eigen::MatrixXd new_meshV1 = meshV1;
        //update rhs with the fixed constraints
        for(int dim = 0; dim < 3; dim++)
        {
            Eigen::VectorXd new_rhs = Eigen::VectorXd::Zero(L_star.rows());
            for(int iv = 0; iv < meshV_.rows(); iv++)
            {
                if(map_fixed_vertices_as_true.find(iv) == map_fixed_vertices_as_true.end())
                {
                    int new_iv = map_nonfixed_vertices_to_new_indices[iv];
                    new_rhs(new_iv) = rhs(iv, dim);
                    for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(L_, iv); it; ++it)
                    {
                        int jv = it.col();
                        if(map_fixed_vertices_as_true.find(jv) != map_fixed_vertices_as_true.end()){
                            double value = it.value();
                            new_rhs(new_iv) -= value * map_fixed_vertices_values[jv][dim];
                        }
                    }
                }
            }

            Eigen::VectorXd result = lu.solve(new_rhs);
//            std::cout << result << std::endl;
            for(int iv = 0; iv < meshV_.rows(); iv++){
                if(map_fixed_vertices_as_true.find(iv) == map_fixed_vertices_as_true.end())
                {
                    int new_iv = map_nonfixed_vertices_to_new_indices[iv];
//                    std::cout << new_iv << " " << result[new_iv] << std::endl;
                    new_meshV1(iv, dim) = result(new_iv);
                }
                else{
                    new_meshV1(iv, dim) = map_fixed_vertices_values[iv][dim];
                }
            }
        }

        meshV1 = new_meshV1;
    }

    return meshV1;
}

void MeshVoxelARAP::compute_energy(const Eigen::MatrixXd &meshV1,
                                   const vector<Eigen::MatrixXd> &Rs,
                                   double &E,
                                   Eigen::MatrixXd &gradient) {
    Eigen::MatrixXd gradientDistance;
    double distance = 0;
    computeDiffDistanceToSelectedVoxels(meshV1, distance, gradientDistance);

    Eigen::MatrixXd gradientShape;
    double shape = 0;
    compute_shape_enegry(meshV1, Rs, shape, gradientShape);

    E = shape * shape_weight_ + distance;
    std::cout << shape << ", " << distance << std::endl;
    gradient = gradientShape * shape_weight_ + gradientDistance;
}