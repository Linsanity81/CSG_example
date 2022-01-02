//
// Created by ziqwang on 28.12.21.
//

#include "SurfaceEvo.h"
#include "SurfaceSlice.h"
#include "catch2/catch_all.hpp"
#include <iostream>
#include "fusion.h"
#include "MeshVoxel.h"
using namespace mosek::fusion;
using namespace monty;

TEST_CASE("SurfaceSliceOpt"){
    vector<double> volumes;

    vector<Eigen::Vector3i> voxel_indices;

    std::shared_ptr<MeshVoxel> meshVoxel;

    vector<double> data_xs = {0,
                              0.3,
                              0.4,
                              0.6,
                              1.1};

    vector<double> data_yts = {0.4, -1,
                               0.3, 0.3,
                               0.45, 0.3,
                               0.45, -0.3,
                               0.3, -0.3};

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    SurfaceEvo surface(data_xs);
    surface.computeMesh(data_yts, V, F);

    double grids_size = 10;
    double grids_width = 0.11;
    Eigen::Vector3d grids_origin = Eigen::Vector3d( 0,-grids_width * grids_size / 2, -grids_width * grids_size / 2);

    meshVoxel = std::make_shared<MeshVoxel>(grids_origin, grids_width, grids_size, 0.3);
    meshVoxel->meshV_ = V;
    meshVoxel->meshF_ = F;
    meshVoxel->voxelization_approximation(volumes, voxel_indices);
    meshVoxel->computeSelectedVoxels(volumes, voxel_indices);

    SurfaceSlice surfaceSlice(0, 1.1);
    surfaceSlice.initSurface(grids_origin, grids_width, grids_size, meshVoxel->selected_voxel_indices_);

    int NX = surfaceSlice.num_x_sample_;
    int NT = surfaceSlice.num_theta_sample_;
    int nvar = NX * NT;
    double weight = 10.0;

    vector<double> radius_upper_bound;
    for(int id = 0; id < surfaceSlice.radius_.size(); id++){
        for(int jd = 0; jd < surfaceSlice.radius_[id].size(); jd++){
            radius_upper_bound.push_back(surfaceSlice.radius_[id][jd]);
        }
    }

    auto mosek_radius_upper_bound = new_array_ptr<double>(radius_upper_bound);

    Model::t M = new Model("quadratic"); auto _M = finally([&]() { M->dispose(); });
    Variable::t var_r  = M->variable("radius", nvar, Domain::inRange(0.0, mosek_radius_upper_bound));
    Variable::t var_g = M->variable("gap", nvar);
    M->constraint("con_gap", Expr::add(var_r, var_g), Domain::equalsTo(mosek_radius_upper_bound));

    Variable::t var_sh = M->variable("smooth_horizontal", nvar);
    Variable::t var_sv = M->variable("smooth_vertical", (NX - 2) * NT);

    int icon = 0;
    for(int id = 0; id < NX; id++)
    {
        for(int jd = 0; jd < NT; jd++)
        {
            int prev_iv = id * NT + (jd - 1 + NT) % NT;
            int next_iv = id * NT + (jd + 1) % NT;
            int iv = id * NT + jd;
            auto expression = Expr::add(var_r->index(prev_iv), var_r->index(next_iv));
            expression = Expr::sub(expression, Expr::mul(2.0, var_r->index(iv)));
            expression = Expr::mul(expression, sqrt(weight));
            expression = Expr::sub(expression, var_sh->index(iv));
            std::string con_str = "contraints_hortizontal_" + std::to_string(icon);
            M->constraint(con_str, expression, Domain::equalsTo(0.0));
            icon++;
        }
    }

    icon = 0;
    for(int id = 1; id + 1 < NX; id++)
    {
        for(int jd = 0; jd < NT; jd++)
        {
            int prev_iv = (id - 1) * NT + jd;
            int next_iv = (id + 1) * NT + jd;
            int iv = id * NT + jd;

            auto expression = Expr::add(var_r->index(prev_iv), var_r->index(next_iv));
            expression = Expr::sub(expression, Expr::mul(2.0, var_r->index(iv)));
            expression = Expr::mul(expression, sqrt(weight));
            expression = Expr::sub(expression, var_sv->index(icon));
            std::string con_str = "contraints_vertical_" + std::to_string(icon);
            M->constraint(con_str, expression, Domain::equalsTo(0.0));
            icon++;
        }
    }

    Variable::t var_obj = M->variable(1);
    std::shared_ptr<ndarray<Variable::t,1>> varlist
            = new_array_ptr<Variable::t,1>({var_obj, var_g, var_sh, var_sv});

    Variable::t stack_var = Var::vstack(varlist);

    M->constraint("obj_contrain", stack_var, Domain::inQCone());

    M->objective("obj", ObjectiveSense::Minimize, var_obj);

    M->solve();

    ndarray<double, 1> var_radius   = *(var_r->level());
    for(int id = 0; id < var_radius.size(); id++){
        std::cout << var_radius[id] << std::endl;
    }
}

//TEST_CASE("Mosek"){
//
//    Model::t M = new Model("cqo1"); auto _M = finally([&]() { M->dispose(); });
//
//    Variable::t x  = M->variable("x", 3, Domain::greaterThan(0.0));
//    Variable::t y  = M->variable("y", 3, Domain::unbounded());
//
//    // Create the aliases
//    //      z1 = [ y[0],x[0],x[1] ]
//    //  and z2 = [ y[1],y[2],x[2] ]
//    Variable::t z1 = Var::vstack(y->index(0),  x->slice(0, 2));
//    Variable::t z2 = Var::vstack(y->slice(1, 3), x->index(2));
//
//    // Create the constraint
//    //      x[0] + x[1] + 2.0 x[2] = 1.0
//    auto aval = new_array_ptr<double, 1>({1.0, 1.0, 2.0});
//    M->constraint("lc", Expr::dot(aval, x), Domain::equalsTo(1.0));
//
//    // Create the constraints
//    //      z1 belongs to C_3
//    //      z2 belongs to K_3
//    // where C_3 and K_3 are respectively the quadratic and
//    // rotated quadratic cone of size 3, i.e.
//    //                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
//    //  and  2.0 z2[0] z2[1] >= z2[2]^2
//    Constraint::t qc1 = M->constraint("qc1", z1, Domain::inQCone());
//    Constraint::t qc2 = M->constraint("qc2", z2, Domain::inRotatedQCone());
//
//    // Set the objective function to (y[0] + y[1] + y[2])
//    M->objective("obj", ObjectiveSense::Minimize, Expr::sum(y));
//
//    // Solve the problem
//    M->solve();
//
//    // Get the linear solution values
//    ndarray<double, 1> xlvl   = *(x->level());
//    ndarray<double, 1> ylvl   = *(y->level());
//    // Get conic solution of qc1
//    ndarray<double, 1> qc1lvl = *(qc1->level());
//    ndarray<double, 1> qc1dl  = *(qc1->dual());
//
//    std::cout << "x1,x2,x2 = " << xlvl << std::endl;
//    std::cout << "y1,y2,y3 = " << ylvl << std::endl;
//    std::cout << "qc1 levels = " << qc1lvl << std::endl;
//    std::cout << "qc1 dual conic var levels = " << qc1dl << std::endl;
//}