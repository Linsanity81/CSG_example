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
};


#endif //MESHVOXEL_MESHVOXELOPT_H
