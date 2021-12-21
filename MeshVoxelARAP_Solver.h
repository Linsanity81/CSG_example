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

    void optimize(Eigen::MatrixXd &meshV1)
    {
        intermediate_results_.clear();
        Eigen::MatrixXd meshV = base_mesh_->meshV_;
        Eigen::VectorXi b;
        base_mesh_->precompute_arap_data(b);

        max_outer_it_time_ = 2;
        max_inner_it_time_ = 5;

        int outer_it;
        while (outer_it < max_outer_it_time_)
        {

            if(max_outer_it_time_ - 1 == outer_it){
                base_mesh_->shape_weight_ = .1;
            }
            else{
                base_mesh_->shape_weight_ = 20;
            }

            vector<double> volumes;
            vector<Eigen::Vector3i> voxel_indices;

            base_mesh_->meshV_ = meshV1;
            base_mesh_->voxelization_approximation(volumes, voxel_indices);
            base_mesh_->computeSelectedVoxels(volumes, voxel_indices);
            base_mesh_->meshV_ = meshV;

            // Set up parameters
            LBFGSpp::LBFGSParam<double> param;
            param.epsilon = 1e-8;
            param.max_iterations = 100;
            param.max_linesearch = 100;

            int inner_it = 0;
            while (inner_it < max_inner_it_time_) {
                base_mesh_->compute_rotation_matrices(meshV1);

                // Create solver and function object
                LBFGSpp::LBFGSSolver<double,
                LBFGSpp::LineSearchBacktracking> solver(param);

                // Initial guess
                Eigen::VectorXd x;
                base_mesh_->flatten(meshV1, x);

                double fx;
                int niter = solver.minimize(*base_mesh_, x, fx);
                base_mesh_->reshape(x, meshV1);

                Eigen::MatrixXd gradient;
                base_mesh_->compute_point_to_selected_voxels_distance(meshV1, fx, gradient);

                inner_it ++;
            }
            intermediate_results_.push_back(meshV1);
            outer_it++;
        }
    }

public:
    MeshVoxelARAP_Solver(std::shared_ptr<MeshVoxelARAP> mesh){
            base_mesh_ = mesh;
    }
};


#endif //MESHVOXEL_MESHVOXELARAP_SOLVER_H
