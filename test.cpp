//
// Created by ziqwang on 17.11.21.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "MeshVoxel.h"
#include <vector>
#include <iostream>
#include "MeshVoxelOpt.h"
#include "MeshVoxelARAP.h"
#include <memory>
#include "igl/writeOBJ.h"
#include "igl/arap.h"

TEST_CASE("Compute Distance"){
    Eigen::Vector3d grids_origin = Eigen::Vector3d(-1, -1, -1);
    double grids_size = 10;
    double grids_width = 2.0 / grids_size;

    std::shared_ptr<MeshVoxelARAP> meshVoxelARAP
            = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);

    meshVoxelARAP->selected_voxel_indices_.push_back(Eigen::Vector3i(3, 3, 3));
    meshVoxelARAP->selected_voxel_indices_.push_back(Eigen::Vector3i(3, 4, 3));
    meshVoxelARAP->selected_voxel_indices_.push_back(Eigen::Vector3i(4, 4, 3));

    Eigen::MatrixXd tv(5, 3);
    tv <<
    0.1, 0.1, 0.1,
    -0.1, -0.1, -0.1,
    -0.15,-0.15, -0.15,
    0, 0, 0,
    0.3, 0.3, 0.3;

    vector<vector<int>> group_pts;
    vector<Eigen::Vector3i> group_voxel_indices;

    meshVoxelARAP->cluster_points_to_voxel_groups(tv, group_pts, group_voxel_indices);

    for(int id = 0; id < group_voxel_indices.size(); id++)
    {
        std::cout << group_voxel_indices[id].transpose() << std::endl;
    }

    for(int id = 0; id < group_pts.size(); id++)
    {
        for(int jd = 0; jd < group_pts[id].size(); jd++){
            std::cout << group_pts[id][jd] << " ";
        }
        std::cout << std::endl;
    }


    vector<double> distances;
    vector<Eigen::Vector3i> sorted_voxel_indices;
    meshVoxelARAP->sort_selected_voxel_given_voxel_group(
            Eigen::Vector3i(5, 5, 5),
            sorted_voxel_indices,
            distances
            );

    for(int id = 0; id < distances.size(); id++){
        std::cout << sorted_voxel_indices[id].transpose()
        << ": " << distances[id] << std::endl;
    }

    double distance;
    Eigen::MatrixXd gradient;
    meshVoxelARAP->compute_point_to_selected_voxels_distance(tv, distance, gradient);






}

//TEST_CASE("Subdivide Triangles")
//{
//    Eigen::MatrixXd base_tri(3, 3);
//    base_tri << 0, 0, 0,
//    1, 0, 0,
//    0, 1, 0;
//    Eigen::MatrixXd curr_baries(3, 3);
//    curr_baries << 1, 0, 0,
//    0, 1, 0,
//    0, 0, 1;
//
//    Eigen::Vector3d grids_origin = Eigen::Vector3d(-1, -1, -1);
//    double grids_size = 10;
//    double grids_width = 2.0 / grids_size;
//
//    std::shared_ptr<MeshVoxelARAP> meshVoxelARAP
//            = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
//
//    std::vector<Eigen::Vector3d> sampled_baries;
//    meshVoxelARAP->subdivide_triangle(base_tri, curr_baries, sampled_baries);
//
//    for(int id = 0; id < sampled_baries.size(); id++){
//        std::cout << sampled_baries[id].transpose() << std::endl;
//    }
//
//}

//TEST_CASE("Mesh"){
//    Eigen::Vector3d grids_origin = Eigen::Vector3d(-1, -1, -1);
//    double grids_size = 10;
//    double grids_width = 2.0 / grids_size;
//
//    std::shared_ptr<MeshVoxelARAP> meshVoxelARAP
//    = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
//
//    std::string filename = "../data/Model/Organic/Duck";
//    meshVoxelARAP->readMesh(filename + ".obj");
//
//    Eigen::MatrixXd meshV = meshVoxelARAP->meshV_;
//    Eigen::VectorXi b(2);
//    b << 0, meshV.rows() - 1;
//    Eigen::MatrixXd bc(2, 3);
//    bc.row(0) = meshV.row(b[0]).transpose() + Eigen::Vector3d(0, 0, 1.0);
//    bc.row(1) = meshV.row(b[1]);
//
//    //meshVoxelARAP->precompute_arap_data(b);
//    //meshV = meshVoxelARAP->solve_arap(bc, 10);
//    meshV = meshVoxelARAP->deform(b, bc, 10);
//    igl::writeOBJ("test.obj", meshV, meshVoxelARAP->meshF_);
//
//    meshV = meshVoxelARAP->meshV_;
//    igl::ARAPData data;
//    igl::arap_precomputation(meshVoxelARAP->meshV_, meshVoxelARAP->meshF_, 3, b, data);
//    data.max_iter = 10;
//    igl::arap_solve(bc, data, meshV);
//
//    igl::writeOBJ("test1.obj", meshV, meshVoxelARAP->meshF_);
//}

//TEST_CASE("Tet"){
//    MeshVoxelOpt meshVoxelOpt(Eigen::Vector3d(0, 0, 0), 1, 5, 0.3);
//    meshVoxelOpt.TT_ = Eigen::MatrixXi(1, 4);
//    meshVoxelOpt.TT_ << 0, 1, 2, 3;
//    meshVoxelOpt.TV_ = Eigen::MatrixXd(4, 3);
//    meshVoxelOpt.TV_ << 0, 0, 0,
//    1, 0, 0,
//    0, 1, 0,
//    0, 0, 1;
//
//    meshVoxelOpt.computeEdgeLength();
//
//    Eigen::MatrixXd tv(4, 3);
//    tv << 0, 0, 0,
//    2, 0, 0,
//    0, 3, 0,
//    0, 0, 4;
//
//    double energy;
//    Eigen::VectorXd gradient;
//    meshVoxelOpt.computeDiffShapeEnergy(tv, energy, gradient);
//
//    for(double id = 3; id < 10; id += 0.2){
//        double eps = pow(10, -id);
//        Eigen::MatrixXd randmat = Eigen::MatrixXd::Random(4, 3);
//
//        Eigen::MatrixXd ntv = tv + randmat * eps;
//
//        double new_energy;
//        Eigen::MatrixXd new_gradient;
//        meshVoxelOpt.computeDiffShapeEnergy(ntv, new_energy, new_gradient);
//        std::cout << eps << ", " << ((new_energy - energy) - (gradient.array() * randmat.array() * eps).sum()) / eps << std::endl;
//    }
//
//    double distance = 0;
//    meshVoxelOpt.computeDiffDistanceToSelectedVoxels(tv,
//                                                     {Eigen::Vector3i(2, 2, 1), Eigen::Vector3i(0, 1, 0)},
//                                                     distance,
//                                                     gradient);
//    std::cout << distance << std::endl;
//    std::cout << gradient << std::endl;
//
//}
//
//TEST_CASE( "Contact Face Computation" ) {
//    MeshVoxel mesh(Eigen::Vector3d(0, 0, 0), 1, 2);
//    Eigen::MatrixXd V(3, 3);
//    Eigen::MatrixXi F(1, 3);
//    V << 0, 0, 0,
//         1, 0, 0,
//         1, 1, 0;
//
//    F << 0, 1, 2;
//
//    std::vector<double> areas
//    = mesh.compute_contacts(Eigen::Vector3i(0, 0, 0), V, F);
//    REQUIRE(areas[5] == Catch::Approx(0.0));
//    REQUIRE(areas[4] == Catch::Approx(0.0));
//
//    V << 0, 0, 0,
//         1, 1, 0,
//         1, 0, 0;
//
//    areas = mesh.compute_contacts(Eigen::Vector3i(0, 0, 0), V, F);
//    REQUIRE(areas[5] == Catch::Approx(0.0));
//    REQUIRE(areas[4] == Catch::Approx(0.5));
//}