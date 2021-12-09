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

TEST_CASE("Mesh"){
    Eigen::Vector3d grids_origin = Eigen::Vector3d(-1, -1, -1);
    double grids_size = 10;
    double grids_width = 2.0 / grids_size;

    std::shared_ptr<MeshVoxelARAP> meshVoxelARAP
    = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);

    meshVoxelARAP->readMesh("../data/plane_dense.obj");
    Eigen::MatrixXd meshV = meshVoxelARAP->meshV_;
    Eigen::VectorXi b(2);
    b << 0, meshV.rows() - 1;
    Eigen::MatrixXd bc(2, 3);
    bc.row(0) = meshV.row(b[0]).transpose() + Eigen::Vector3d(0, 0, 1.0);
    bc.row(1) = meshV.row(b[1]);

    meshV = meshVoxelARAP->deform(b, bc, 10);
    igl::writeOBJ("test.obj", meshV, meshVoxelARAP->meshF_);

    meshV = meshVoxelARAP->meshV_;
    igl::ARAPData data;
    igl::arap_precomputation(meshVoxelARAP->meshV_, meshVoxelARAP->meshF_, 3, b, data);
    igl::arap_solve(bc, data, meshV);

    igl::writeOBJ("test1.obj", meshV, meshVoxelARAP->meshF_);
}

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