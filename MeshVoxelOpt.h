//
// Created by ziqwang on 30.11.21.
//

#ifndef MESHVOXEL_MESHVOXELOPT_H
#define MESHVOXEL_MESHVOXELOPT_H

#include "MeshVoxel.h"
#include <map>
#include <vector>
#include <iostream>
using std::vector;

class MeshVoxelOpt: public MeshVoxel{
public:
    Eigen::MatrixXd TV_;

    Eigen::MatrixXi TT_;

    Eigen::MatrixXi TF_;

    Eigen::MatrixXi E_;

    vector<double> l0;



public:

    double weight_shape_energy;

public:

    MeshVoxelOpt(Eigen::Vector3d ori, double width, int size, double ratio)
    : MeshVoxel(ori, width, size, ratio){
        weight_shape_energy = 1000.0;
    }

public:

    double computeTetVolume(int tetID);

    void computeEdgeLength();

    Eigen::Vector3i point_to_voxel_index(Eigen::Vector3d pt) const;

    void readMesh(std::string filename, std::string tetgen_str);

    void approxVoxelization(vector<Eigen::MatrixXd> &Vs,
                            vector<Eigen::MatrixXi> &Fs,
                            vector<double> &volumes,
                            vector<Eigen::Vector3i> &voxel_indices);

    void computeSelectedVoxels(vector<double> &volumes, vector<Eigen::Vector3i> &voxel_indices);

    void computeDiffShapeEnergy(const Eigen::MatrixXd &tv,
                                double &energy,
                                Eigen::MatrixXd &gradient) const;

    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const;

    void flatten(const Eigen::MatrixXd &mat, Eigen::VectorXd &vec) const{
        vec = Eigen::VectorXd::Zero(mat.rows() * 3);
        for(int id = 0; id < mat.rows(); id++){
            vec[3 * id] = mat(id, 0);
            vec[3 * id + 1] = mat(id, 1);
            vec[3 * id + 2] = mat(id, 2);
        }
    };

    void reshape(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat) const{
        mat = Eigen::MatrixXd(vec.size() / 3, 3);
        for(int id = 0; id < vec.rows() / 3; id++){
            mat(id, 0) = vec[3 * id];
            mat(id, 1) = vec[3 * id + 1];
            mat(id, 2) = vec[3 * id + 2];
        }
    }
};


#endif //MESHVOXEL_MESHVOXELOPT_H
