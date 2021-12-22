//
// Created by ziqwang on 16.12.21.
//

#ifndef MESHVOXEL_MESHVOXELARAP_SOLVER_H
#define MESHVOXEL_MESHVOXELARAP_SOLVER_H

#include "MeshVoxelARAP.h"
#include "LBFGS.h"

class MeshVoxelARAP_Solver{
public:

    std::shared_ptr<MeshVoxelARAP> base_mesh_;

    int max_outer_it_time_;

    int max_inner_it_time_;

    vector<Eigen::MatrixXd> intermediate_results_;

public:

    Eigen::Vector3d optimize_location(Eigen::Vector3d base_origin){
        int nsample = 5;
        Eigen::Vector3d optimal_origin(0, 0, 0);
        double optimal_ratio = 0;

        for(int ix = 0; ix < nsample; ix++)
        {
            for(int iy = 0; iy < nsample; iy++)
            {
                for(int iz = 0; iz < nsample; iz++)
                {
                    Eigen::Vector3d delta;
                    delta(0) = (double) ix / nsample * base_mesh_->grids_width_;
                    delta(1) = (double) iy / nsample * base_mesh_->grids_width_;
                    delta(2) = (double) iz / nsample * base_mesh_->grids_width_;
                    base_mesh_->grids_origin_ = base_origin - delta;
                    vector<double> volumes;
                    vector<Eigen::Vector3i> voxel_indices;
                    base_mesh_->voxelization_approximation(volumes, voxel_indices);
                    double ratio = (double)base_mesh_->computePartialFullnTinyVoxels(volumes) / volumes.size();
                    std::cout << base_mesh_->grids_origin_.transpose() << ", " << ratio << std::endl;
                    if(ratio > optimal_ratio){
                        optimal_origin = base_origin - delta;
                        optimal_ratio = ratio;
                    }
                }
            }
        }

        return optimal_origin;
    }

    void optimize(Eigen::MatrixXd &opt_meshV, Eigen::Vector3d &opt_grids_origin)
    {
        intermediate_results_.clear();

        Eigen::MatrixXd meshV = base_mesh_->meshV_;
        Eigen::VectorXi b;
        base_mesh_->precompute_arap_data(b);

        max_outer_it_time_ = 3;
        max_inner_it_time_ = 3;

        Eigen::Vector3d base_origin = base_mesh_->grids_origin_;

        int outer_it = 0;
        while (outer_it < max_outer_it_time_)
        {
            base_mesh_->meshV_ = opt_meshV;

            //update origin
            opt_grids_origin = optimize_location(base_origin);
            base_mesh_->grids_origin_ = opt_grids_origin;

            //update selected voxels
            vector<double> volumes;
            vector<Eigen::Vector3i> voxel_indices;
            base_mesh_->voxelization_approximation(volumes, voxel_indices);
            base_mesh_->computeSelectedVoxels(volumes, voxel_indices);
            base_mesh_->meshV_ = meshV;

            // Set up parameters
            LBFGSpp::LBFGSParam<double> param;
            param.epsilon = 1e-4;
            param.max_iterations = 50;
            //param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
            param.max_linesearch = 100;

            int inner_it = 0;
            base_mesh_->shape_weight_ = 1E-2;

            while (inner_it < max_inner_it_time_)
            {

                base_mesh_->compute_rotation_matrices(opt_meshV);

                // Create solver and function object
                LBFGSpp::LBFGSSolver<double,
                LBFGSpp::LineSearchBacktracking> solver(param);

                // Initial guess
                Eigen::VectorXd x;
                base_mesh_->flatten(opt_meshV, x);

                double fx;
                int niter = solver.minimize(*base_mesh_, x, fx);
                base_mesh_->reshape(x, opt_meshV);

                Eigen::MatrixXd gradient;
                base_mesh_->compute_point_to_selected_voxels_distance(opt_meshV, fx, gradient);

                //base_mesh_->shape_weight_ /= 10;

                inner_it ++;
            }
            intermediate_results_.push_back(opt_meshV);
            outer_it++;
        }

        base_mesh_->grids_origin_ = base_origin;
    }

public:
    MeshVoxelARAP_Solver(std::shared_ptr<MeshVoxelARAP> mesh){
            base_mesh_ = mesh;
    }
};


#endif //MESHVOXEL_MESHVOXELARAP_SOLVER_H
