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

    int num_location_sample_;

    double shape_weight_;

    double gap_;

    double shape_weight_last_iteration_;

    double lbfgs_eps_;

    int lbfgs_iterations_;

    int lbfgs_linesearch_iterations_;

    vector<Eigen::MatrixXd> intermediate_results_;

    double full_voxel_ratio_;

    double core_voxel_ratio_;

    bool use_location_optimization_;

public:

    Eigen::Vector3d optimize_location(Eigen::Vector3d base_origin){
        int nsample = num_location_sample_;
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
                    double ratio = (double)base_mesh_->computePartialFullnTinyVoxels(volumes, full_voxel_ratio_) / volumes.size();
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

        base_mesh_->gap_ = gap_;

        Eigen::MatrixXd meshV = base_mesh_->meshV_;
        Eigen::VectorXi b;
        base_mesh_->precompute_arap_data(b);

        opt_grids_origin = base_mesh_->grids_origin_;
        Eigen::Vector3d base_origin = base_mesh_->grids_origin_;

        int outer_it = 0;
        while (outer_it < max_outer_it_time_)
        {
            base_mesh_->meshV_ = opt_meshV;

            if(outer_it == max_outer_it_time_ - 1){
                base_mesh_->shape_weight_ = shape_weight_last_iteration_;
            }
            else{
                base_mesh_->shape_weight_ = shape_weight_;
            }

            //update origin
            if(use_location_optimization_){
                opt_grids_origin = optimize_location(base_origin);
                base_mesh_->grids_origin_ = opt_grids_origin;
            }

            //update selected voxels
            vector<double> volumes;
            vector<Eigen::Vector3i> voxel_indices;
            base_mesh_->voxelization_approximation(volumes, voxel_indices);
            base_mesh_->computeSelectedVoxels(volumes, voxel_indices, core_voxel_ratio_);
            base_mesh_->meshV_ = meshV;

            base_mesh_->core_voxels = base_mesh_->selected_voxel_indices_;
            base_mesh_->expansion_voxels(base_mesh_->core_voxels, base_mesh_->boundary_voxels);

            // Set up parameters
            LBFGSpp::LBFGSParam<double> param;
            param.epsilon = lbfgs_eps_;
            param.max_iterations = lbfgs_iterations_;
            //param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
            param.max_linesearch = lbfgs_linesearch_iterations_;

            int inner_it = 0;

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
        num_location_sample_ = 5;
        max_outer_it_time_ = 2;
        max_inner_it_time_ = 3;
        shape_weight_ = 20.0;
        shape_weight_last_iteration_ = 0.1;
        lbfgs_eps_ = 1E-8;
        lbfgs_iterations_ = 50;
        lbfgs_linesearch_iterations_ = 100;
    }
};


#endif //MESHVOXEL_MESHVOXELARAP_SOLVER_H
