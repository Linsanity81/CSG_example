//
// Created by ziqwang on 29.12.21.
//

#include "SurfaceSlice.h"
#include <map>
#include "igl/ray_mesh_intersect.h"

void SurfaceSlice::initSurface(Eigen::Vector3d grids_origin,
                               double grids_width,
                               int grids_size,
                               const vector<Eigen::Vector3i> &voxel_indices,
                               const vector<double> &xs)
                               {

    xs_ = xs;

    std::map<int, bool> selected_voxels;

    for(int id = 0; id < voxel_indices.size(); id++){
        Eigen::Vector3i index = voxel_indices[id];
        int digit = index(0) + index(1) * grids_size + index(2) * grids_size * grids_size;
        selected_voxels[digit] = true;
    }

    int num_angle = 32;

    radius_.resize(xs.size());
    for(int id = 0; id < xs.size(); id++){
        radius_[id].resize(num_angle, std::numeric_limits<double>::max());
    }


    for(int digit = 0; digit < grids_size * grids_size * grids_size; digit++)
    {
        if(selected_voxels[digit] == false){
            int ix = digit % grids_size;
            int iy = ((digit - ix) / grids_size) % grids_size;
            int iz = (digit - ix - iy * grids_size) / (grids_size * grids_size);
            Eigen::Vector3i index(ix, iy, iz);
            double x0 = ix * grids_width + grids_origin(0);
            double x1 = x0 + grids_width;

            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            compute_voxel(index, grids_origin, grids_width, V,F);
            Eigen::Vector3d pt(x0 + x1, 0, 0);
            vector<double> voxel_radius;

            for(int id = 0; id < num_angle; id++){
                double angle = 2 * M_PI / num_angle * id;
                Eigen::Vector3d drt(0, cos(angle), sin(angle));
                igl::Hit hit;
                if(igl::ray_mesh_intersect(pt, drt, V, F, hit)){
                    voxel_radius.push_back(hit.t);
                }
                else{
                    voxel_radius.push_back(std::numeric_limits<double>::max());
                }
            }

            for(int id = 0; id < xs.size(); id++)
            {
                if(xs[id] >= x0 && xs[id] <= x1)
                {
                    for(int jd = 0; jd < num_angle; jd++){
                        radius_[id][jd] = std::min(radius_[id][jd], voxel_radius[jd]);
                    }
                }

                if(xs[id] > x1){
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