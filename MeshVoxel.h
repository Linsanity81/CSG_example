//
// Created by ziqwang on 15.11.21.
//

#ifndef EXAMPLE_MESHVOXEL_H
#define EXAMPLE_MESHVOXEL_H
#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/volume.h>
class MeshVoxel {
public:

    Eigen::MatrixXd meshV_;

    Eigen::MatrixXi meshF_;

    Eigen::Vector3d grids_origin_;

    double grids_width_;

public:

    MeshVoxel(Eigen::Vector3d ori, double width){
        grids_origin_ = ori;
        grids_width_ = width;
    }

public:

    void readMesh(std::string filename);

public:

    double intersec(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

    void voxel(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F);
};


#endif //EXAMPLE_MESHVOXEL_H
