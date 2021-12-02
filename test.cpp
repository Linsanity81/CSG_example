//
// Created by ziqwang on 17.11.21.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "MeshVoxel.h"
#include <vector>
#include <iostream>
#include "MeshVoxelOpt.h"

TEST_CASE("Tet"){
    MeshVoxelOpt meshVoxelOpt(Eigen::Vector3d(0, 0, 0), 1, 5, 0.3);
    meshVoxelOpt.TT_ = Eigen::MatrixXi(1, 4);
    meshVoxelOpt.TT_ << 0, 1, 2, 3;
    meshVoxelOpt.TV_ = Eigen::MatrixXd(4, 3);
    meshVoxelOpt.TV_ << 0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;

    meshVoxelOpt.computeEdgeLength();

    Eigen::MatrixXd tv(4, 3);
    tv << 0, 0, 0,
    2, 0, 0,
    0, 3, 0,
    0, 0, 4;

    double energy;
    Eigen::MatrixXd gradient;
    meshVoxelOpt.computeDiffShapeEnegry(tv, energy, gradient);

    for(double id = 3; id < 10; id += 0.2){
        double eps = pow(10, -id);
        Eigen::MatrixXd randmat = Eigen::MatrixXd::Random(4, 3);

        Eigen::MatrixXd ntv = tv + randmat * eps;

        double new_energy;
        Eigen::MatrixXd new_gradient;
        meshVoxelOpt.computeDiffShapeEnegry(ntv, new_energy, new_gradient);
        std::cout << eps << ", " << ((new_energy - energy) - (gradient.array() * randmat.array() * eps).sum()) / eps << std::endl;
    }

    double distance = 0;
    meshVoxelOpt.computeDiffDistanceToSelectedVoxels(tv, {Eigen::Vector3i(2, 2, 1), Eigen::Vector3i(0, 1, 0)}, distance, gradient);
    std::cout << distance << std::endl;
    std::cout << gradient << std::endl;

}

TEST_CASE( "Contact Face Computation" ) {
    MeshVoxel mesh(Eigen::Vector3d(0, 0, 0), 1, 2);
    Eigen::MatrixXd V(3, 3);
    Eigen::MatrixXi F(1, 3);
    V << 0, 0, 0,
         1, 0, 0,
         1, 1, 0;

    F << 0, 1, 2;

    std::vector<double> areas
    = mesh.compute_contacts(Eigen::Vector3i(0, 0, 0), V, F);
    REQUIRE(areas[5] == Catch::Approx(0.0));
    REQUIRE(areas[4] == Catch::Approx(0.0));

    V << 0, 0, 0,
         1, 1, 0,
         1, 0, 0;

    areas = mesh.compute_contacts(Eigen::Vector3i(0, 0, 0), V, F);
    REQUIRE(areas[5] == Catch::Approx(0.0));
    REQUIRE(areas[4] == Catch::Approx(0.5));
}