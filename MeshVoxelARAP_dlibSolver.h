//
// Created by ziqwang on 10.01.22.
//

#ifndef MESHVOXEL_MESHVOXELARAP_DLIBSOLVER_H
#define MESHVOXEL_MESHVOXELARAP_DLIBSOLVER_H

#include "MeshVoxelARAP.h"
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

typedef dlib::matrix<double,0,1> column_vector;

static void convert_from_dlib_to_eigen(const column_vector &input,
                                  Eigen::VectorXd &output){
    output = Eigen::VectorXd(input.nr());
    for(int id = 0; id < input.nr(); id++){
        output(id) = input(id, 0);
    }
    return;
}

static void convert_from_eigen_to_dlib(const Eigen::VectorXd &input,
                                       column_vector &output){
    output = column_vector (input.rows(), 1);
    for(int id = 0; id < input.rows(); id++){
        output(id, 0) = input(id);
    }
    return;
}


class MeshVoxelARAP_dlib{
public:
    std::shared_ptr<MeshVoxelARAP> meshARAP;

public:

    double operator() (
            const column_vector& x
    ) const {
        Eigen::VectorXd eigen_x;
        convert_from_dlib_to_eigen(x, eigen_x);
        Eigen::VectorXd gradient;
        return meshARAP->operator()(eigen_x, gradient);
    }
};

class MeshVoxelARAP_derivatives_dlib{
public:
    std::shared_ptr<MeshVoxelARAP> meshARAP;

public:

    column_vector operator() (
            const column_vector& x
    ) const {
        Eigen::VectorXd eigen_x;
        convert_from_dlib_to_eigen(x, eigen_x);
        Eigen::VectorXd gradient_eigen;
        meshARAP->operator()(eigen_x, gradient_eigen);
        column_vector gradient;
        convert_from_eigen_to_dlib(gradient_eigen, gradient);
        return gradient;
    }
};

#endif //MESHVOXEL_MESHVOXELARAP_DLIBSOLVER_H
