//
// Created by ziqwang on 15.11.21.
//

#ifndef EXAMPLE_MESHVOXEL_H
#define EXAMPLE_MESHVOXEL_H
#include <Eigen/Dense>
#include <igl/readOBJ.h>
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

    vector<Eigen::Vector3i> selected_voxel_indices_;

public:
    MeshVoxel(Eigen::Vector3d ori, double width, int size){
        grids_origin_ = ori;
        grids_width_ = width;
        grids_size_ = size;
    }

public:

    void readMesh(std::string filename);

public:

    Eigen::Vector3i digit_to_index(int digit) const{
        int ix = digit % grids_size_;
        int iy = ((digit - ix) / grids_size_) % grids_size_;
        int iz = (digit - ix - iy * grids_size_) / (grids_size_ * grids_size_);
        return Eigen::Vector3i(ix, iy, iz);
    }

    int index_to_digit(Eigen::Vector3i index) const
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

    void write_voxels(std::string filename);

    void write_voxels_full_volume(std::string filename);

    void voxelization_approximation_with_empty_voxels(vector<double> &volumes,
                                                      vector<Eigen::Vector3i> &voxel_indices);

    void voxelization_approximation(vector<double> &volumes,
                                    vector<Eigen::Vector3i> &voxel_indices, bool isNeedTinyVoxel = true);

    void computeSelectedVoxels(vector<double> &volumes, vector<Eigen::Vector3i> &voxel_indices, double ratio);

    int computePartialFullnTinyVoxels(vector<double> &volumes, double ratio);

    double computeDistanceVoxelToVoxel(Eigen::Vector3i voxelA, Eigen::Vector3i voxelB) const;

    void expansion_voxels(const vector<Eigen::Vector3i> &input_voxels,
                            vector<Eigen::Vector3i> &expension_voxels);

    Eigen::Vector3i point_to_voxel_index(Eigen::Vector3d pt) const;

    void computeDiffDistancePointToVoxel(Eigen::Vector3d pt,
                                         Eigen::Vector3i voxel_index,
                                         double &distance,
                                         Eigen::Vector3d &gradient) const;

    void cluster_points_to_voxel_groups(const Eigen::MatrixXd &tv,
                                        vector<vector<int>> &group_pts,
                                        vector<Eigen::Vector3i> &group_voxel_indices) const;

    void sort_input_voxels_respect_to_distance_to_given_voxel_group(Eigen::Vector3i voxel_group_index,
                                                                    const vector<Eigen::Vector3i> &input_voxels,
                                                                    vector<Eigen::Vector3i> &sorted_selected_voxels,
                                                                    vector<double> &distance) const;

    void compute_point_to_selected_voxels_distance(const Eigen::MatrixXd &tv,
                                                   double &distance,
                                                   Eigen::MatrixXd &gradient) const;

    void compute_point_to_voxels_distance(const Eigen::MatrixXd &tv,
                                          const vector<Eigen::Vector3i> &boundary_voxels,
                                          const vector<Eigen::Vector3i> &core_voxels,
                                          double tolerance,
                                          double &distance,
                                          Eigen::MatrixXd &gradient) const;

    void compute_triangle_to_voxels_distance(const Eigen::MatrixXd &meshV1,
                                        const vector<Eigen::Vector3i> &boundary_voxels,
                                        const vector<Eigen::Vector3i> &core_voxels,
                                        double tolerance,
                                        double &distance,
                                        Eigen::MatrixXd &gradient) const;

    void compute_triangle_to_selected_voxels_distance(const Eigen::MatrixXd &meshV1,
                                                      double &distance,
                                                      Eigen::MatrixXd &gradient) const;

    void compute_voxel(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

    void subdivide_triangle(int faceID,
                            const Eigen::MatrixXd& meshV1,
                            Eigen::MatrixXd curr_tri_bary_coords,
                            vector<Eigen::Vector3d> &bary_coords,
                            vector<int> &bary_coords_faceID) const;

    void flatten(const Eigen::MatrixXd &mat, Eigen::VectorXd &vec) const;

    void reshape(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat) const;
};


#endif //EXAMPLE_MESHVOXEL_H
