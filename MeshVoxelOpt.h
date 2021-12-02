//
// Created by ziqwang on 30.11.21.
//

#ifndef MESHVOXEL_MESHVOXELOPT_H
#define MESHVOXEL_MESHVOXELOPT_H

#include "MeshVoxel.h"
#include <map>

class MeshVoxelOpt: public MeshVoxel{
public:
    Eigen::MatrixXd TV_;

    Eigen::MatrixXi TT_;

    Eigen::MatrixXi TF_;

    Eigen::MatrixXi E_;

    vector<double> l0;

    double minimum_volume_;

public:

public:

    MeshVoxelOpt(Eigen::Vector3d ori, double width, int size, double ratio)
    : MeshVoxel(ori, width, size){
        minimum_volume_ = width * width * width * ratio;
    }

public:

    double computeTetVolume(int tetID);

    void computeEdgeLength();

    Eigen::Vector3i point_to_voxel_index(Eigen::Vector3d pt);

    void readMesh(std::string filename, std::string tetgen_str);

    void approxVoxelization(vector<Eigen::MatrixXd> &Vs,
                            vector<Eigen::MatrixXi> &Fs,
                            vector<double> &volumes,
                            vector<Eigen::Vector3i> &voxel_indices);

    void computeSelectedVoxels(vector<double> &selected_voxel_volumes,
                               vector<Eigen::Vector3i> &selected_voxel_indices);

    int computeDistanceVoxelToVoxel(Eigen::Vector3i voxelA, Eigen::Vector3i voxelB);

    void computeDiffDistancePointToVoxel(Eigen::Vector3d pt,
                                         Eigen::Vector3i voxel_index,
                                         double &distance,
                                         Eigen::Vector3d &gradient);

    void computeDiffDistanceToSelectedVoxels(const Eigen::MatrixXd &tv,
                                             const vector<Eigen::Vector3i> &selected_voxel_indices,
                                             double &distance,
                                             Eigen::MatrixXd &gradient);

    void computeDiffShapeEnegry(const Eigen::MatrixXd &tv,
                                double &energy,
                                Eigen::MatrixXd &gradient);

};


#endif //MESHVOXEL_MESHVOXELOPT_H
