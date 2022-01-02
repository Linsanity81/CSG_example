//
// Created by ziqwang on 28.12.21.
//

#include "SurfaceEvo.h"

#include "catch2/catch_all.hpp"
#include "MeshVoxelARAP.h"


#include <iostream>
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

TEST_CASE("Mosek"){
    
    Model::t M = new Model("cqo1"); auto _M = finally([&]() { M->dispose(); });

    Variable::t x  = M->variable("x", 3, Domain::greaterThan(0.0));
    Variable::t y  = M->variable("y", 3, Domain::unbounded());

    // Create the aliases
    //      z1 = [ y[0],x[0],x[1] ]
    //  and z2 = [ y[1],y[2],x[2] ]
    Variable::t z1 = Var::vstack(y->index(0),  x->slice(0, 2));
    Variable::t z2 = Var::vstack(y->slice(1, 3), x->index(2));

    // Create the constraint
    //      x[0] + x[1] + 2.0 x[2] = 1.0
    auto aval = new_array_ptr<double, 1>({1.0, 1.0, 2.0});
    M->constraint("lc", Expr::dot(aval, x), Domain::equalsTo(1.0));

    // Create the constraints
    //      z1 belongs to C_3
    //      z2 belongs to K_3
    // where C_3 and K_3 are respectively the quadratic and
    // rotated quadratic cone of size 3, i.e.
    //                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
    //  and  2.0 z2[0] z2[1] >= z2[2]^2
    Constraint::t qc1 = M->constraint("qc1", z1, Domain::inQCone());
    Constraint::t qc2 = M->constraint("qc2", z2, Domain::inRotatedQCone());

    // Set the objective function to (y[0] + y[1] + y[2])
    M->objective("obj", ObjectiveSense::Minimize, Expr::sum(y));

    // Solve the problem
    M->solve();

    // Get the linear solution values
    ndarray<double, 1> xlvl   = *(x->level());
    ndarray<double, 1> ylvl   = *(y->level());
    // Get conic solution of qc1
    ndarray<double, 1> qc1lvl = *(qc1->level());
    ndarray<double, 1> qc1dl  = *(qc1->dual());

    std::cout << "x1,x2,x2 = " << xlvl << std::endl;
    std::cout << "y1,y2,y3 = " << ylvl << std::endl;
    std::cout << "qc1 levels = " << qc1lvl << std::endl;
    std::cout << "qc1 dual conic var levels = " << qc1dl << std::endl;
}
//TEST_CASE("Surface Evo")
//{
//    vector<double> data_xs = {0, 0.5, 1, 1.5,  2};
//
//    vector<double> data_yts = {0.5, 0, 1, 0, 0.5, 0, 1, 0, 0.5, 0};
//
//    Eigen::MatrixXd V;
//    Eigen::MatrixXi F;
//
//    SurfaceEvo surface(data_xs);
//
//    surface.computeMesh(data_yts, V, F);
//
//    double grids_size = 10;
//    double grids_width = 0.25;
//    Eigen::Vector3d grids_origin = Eigen::Vector3d( 0,-grids_width * grids_size / 2, -grids_width * grids_size / 2);
//
//    std::shared_ptr<MeshVoxelARAP> meshVoxelArap;
//
//    vector<double> volumes;
//    vector<Eigen::Vector3i> voxel_indices;
//
//    meshVoxelArap = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
//    meshVoxelArap->meshV_ = V;
//    meshVoxelArap->meshF_ = F;
//    meshVoxelArap->voxelization_approximation(volumes, voxel_indices);
//    meshVoxelArap->computeSelectedVoxels(volumes, voxel_indices);
//    std::cout << meshVoxelArap->selected_voxel_indices_.size() / (double)volumes.size() << std::endl;
//
//    vector<double> xs;
//    int num_sample = 100;
//    for(int id = 0; id < num_sample; id++){
//        xs.push_back(2.0 / num_sample * id);
//    }
//
//    vector<double> radius;
//    surface.computeRadius(grids_origin,
//                          grids_width,
//                          grids_size,
//                          meshVoxelArap->selected_voxel_indices_,
//                          xs,
//                          radius);
//
//
//    Eigen::MatrixXd Mat;
//    Eigen::VectorXd b;
//    surface.compute_constraints(xs, radius, Mat, b);
//
//    Eigen::VectorXd var(data_yts.size());
//    for(int id = 0; id < data_yts.size(); id++) var[id] = data_yts[id];
//
//    // Create a problem instance.
//    SurfaceShrink instance = SurfaceShrink(data_xs, data_yts, xs, Mat, b);
//
//    // Create a solver
//    knitro::KNSolver solver(&instance);
//
//    solver.initProblem();
//    int solveStatus = solver.solve();
//
//    std::vector<double> x;
//    std::vector<double> lambda;
//    int nStatus = solver.getSolution(x, lambda);
//
//    std::vector<double> con = solver.getConstraintValues();
//
//    for(int id = 0; id < data_yts.size(); id++){
//        std::cout << data_yts[id] << " " << x[id] << std::endl;
//    }
//
//}