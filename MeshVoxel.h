//
// Created by ziqwang on 15.11.21.
//

#ifndef EXAMPLE_MESHVOXEL_H
#define EXAMPLE_MESHVOXEL_H
#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/volume.h>
#include <vector>
using std::vector;
class MeshVoxel {
public:
    const int dX[6] = {-1, 1, 0, 0, 0, 0};
    const int dY[6] = {0, 0, -1, 1, 0, 0};
    const int dZ[6] = {0, 0, 0, 0, -1, 1};

public:

    Eigen::MatrixXd meshV_;

    Eigen::MatrixXi meshF_;

    Eigen::Vector3d grids_origin_;

    double grids_width_;

    int grids_size_;

public:

    MeshVoxel(Eigen::Vector3d ori, double width, int size){
        grids_origin_ = ori;
        grids_width_ = width;
        grids_size_ = size;
    }

public:

    void readMesh(std::string filename);

public:

    Eigen::Vector3i digit_to_index(int digit){
        int ix = digit % grids_size_;
        int iy = ((digit - ix) / grids_size_) % grids_size_;
        int iz = (digit - ix - iy * grids_size_) / (grids_size_ * grids_size_);
        return Eigen::Vector3i(ix, iy, iz);
    }

    int index_to_digit(Eigen::Vector3i index)
    {
        int ix = index[0];
        int iy = index[1];
        int iz = index[2];
        if(ix >= 0 && ix < grids_size_ && iy >= 0 && iy < grids_size_ && iz >= 0 && iz < grids_size_){
            return ix + iy * grids_size_ + iz * grids_size_ * grids_size_;
        }
        else{
            return -1;
        }

    }

    void voxelization(vector<Eigen::MatrixXd> &Vs,
                      vector<Eigen::MatrixXi> &Fs,
                      vector<double> &volumes,
                      vector<vector<double>> &areas,
                      vector<Eigen::Vector3i> &voxel_indices);

    std::vector<double> compute_contacts(Eigen::Vector3i index,
                                         const Eigen::MatrixXd &V,
                                         const Eigen::MatrixXi &F);

    double compute_intersec(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

    void compute_voxel(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F);
};


#endif //EXAMPLE_MESHVOXEL_H
