//
// Created by ziqwang on 28.12.21.
//

#include "SurfaceEvo.h"

#include "catch2/catch_all.hpp"
#include "MeshVoxelARAP.h"

TEST_CASE("Surface Evo")
{
    vector<double> data_xs = {0, 0.5, 1, 1.5,  2};

    vector<double> data_yts = {0.5, 0, 1, 0, 0.5, 0, 1, 0, 0.5, 0};

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    SurfaceEvo surface(data_xs);

    surface.computeMesh(data_yts, V, F);

    double grids_size = 10;
    double grids_width = 0.25;
    Eigen::Vector3d grids_origin = Eigen::Vector3d( 0,-grids_width * grids_size / 2, -grids_width * grids_size / 2);

    std::shared_ptr<MeshVoxelARAP> meshVoxelArap;

    vector<double> volumes;
    vector<Eigen::Vector3i> voxel_indices;

    meshVoxelArap = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
    meshVoxelArap->meshV_ = V;
    meshVoxelArap->meshF_ = F;
    meshVoxelArap->voxelization_approximation(volumes, voxel_indices);
    meshVoxelArap->computeSelectedVoxels(volumes, voxel_indices);
    std::cout << meshVoxelArap->selected_voxel_indices_.size() / (double)volumes.size() << std::endl;

    vector<double> xs;
    int num_sample = 100;
    for(int id = 0; id < num_sample; id++){
        xs.push_back(2.0 / num_sample * id);
    }

    vector<double> radius;
    surface.computeRadius(grids_origin,
                          grids_width,
                          grids_size,
                          meshVoxelArap->selected_voxel_indices_,
                          xs,
                          radius);


    Eigen::MatrixXd Mat;
    Eigen::VectorXd b;
    surface.compute_constraints(xs, radius, Mat, b);

    Eigen::VectorXd var(data_yts.size());
    for(int id = 0; id < data_yts.size(); id++) var[id] = data_yts[id];

    // Create a problem instance.
    SurfaceShrink instance = SurfaceShrink(data_xs, data_yts, xs, Mat, b);

    // Create a solver
    knitro::KNSolver solver(&instance);

    solver.initProblem();
    int solveStatus = solver.solve();

    std::vector<double> x;
    std::vector<double> lambda;
    int nStatus = solver.getSolution(x, lambda);

    std::vector<double> con = solver.getConstraintValues();

    for(int id = 0; id < data_yts.size(); id++){
        std::cout << data_yts[id] << " " << x[id] << std::endl;
    }

}