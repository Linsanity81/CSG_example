//
// Created by ziqwang on 17.11.21.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "MeshVoxel.h"
#include <vector>
#include <iostream>

TEST_CASE( "Contact Face Computation" ) {
    MeshVoxel mesh(Eigen::Vector3d(0, 0, 0), 1);
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