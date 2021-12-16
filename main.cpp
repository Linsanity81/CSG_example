#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/edges.h>
#include <vector>
#include "MeshVoxel.h"
#include "MeshVoxelARAP.h"
#include "LBFGS.h"
#include <igl/writeOBJ.h>
#include <filesystem>
#include "MeshVoxelARAP_Solver.h"
#include "knitro.h"
using std::vector;

vector<Eigen::MatrixXd> Vs;

vector<Eigen::MatrixXi> Fs;

vector<double> volumes;

vector<vector<double>> areas;

vector<Eigen::Vector3i> voxel_indices;

std::shared_ptr<MeshVoxelARAP> meshVoxelArap;

void add_edges(igl::opengl::glfw::Viewer& viewer)
{
    for(int id = 0; id < meshVoxelArap->selected_voxel_indices_.size(); id++)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        meshVoxelArap->compute_voxel(meshVoxelArap->selected_voxel_indices_[id], V, F);

        Eigen::MatrixXi E;
        igl::edges(F, E);

        Eigen::MatrixXd P1(E.rows(), 3);
        Eigen::MatrixXd P2(E.rows(), 3);
        for(int jd = 0; jd < E.rows(); jd++){
            P1.row(jd) = V.row(E(jd, 0));
            P2.row(jd) = V.row(E(jd, 1));
        }

        viewer.data().add_edges(P1, P2, Eigen::RowVector3d(1, 1, 1));
    }
}


// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    using namespace std;
    using namespace Eigen;

    if (key >= '1' && key <= '9')
    {
        double t = double((key - '1')+1);
        vector<Eigen::Vector3d> vers;
        vector<Eigen::Vector3i> faces;
        int num_vers = 0;
        for (unsigned i=0; i<voxel_indices.size();++i)
        {
            int iz = voxel_indices[i][2];
            double r_iz = (double)iz;
            if(r_iz < t){
                for(int jd = 0; jd < Vs[i].rows(); jd++){
                    vers.push_back(Vs[i].row(jd));
                }
                for(int jd = 0; jd < Fs[i].rows(); jd++){
                    faces.push_back(Fs[i].row(jd) + Eigen::RowVector3i(num_vers, num_vers, num_vers));
                }
            }
            num_vers = vers.size();
        }

        MatrixXd V_temp(vers.size(),3);
        MatrixXi F_temp(faces.size(),3);

        for (unsigned i=0; i<vers.size();++i)
        {
            V_temp.row(i) = vers[i];
        }

        for (unsigned i=0; i<faces.size();++i)
        {
            F_temp.row(i) = faces[i];
        }

        viewer.data().clear();
        viewer.data().set_mesh(V_temp,F_temp);
        viewer.data().set_face_based(true);
        add_edges(viewer);

    }
    return false;
}

int main() {
    Eigen::Vector3d grids_origin = Eigen::Vector3d(-1, -1, -1.2);
    double grids_size = 8;
    double grids_width = 2.0 / grids_size;

    meshVoxelArap = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
    std::string filename = "../data/Model/Organic/Squirrel";
//    std::string filename = "../test1_0";
    meshVoxelArap->readMesh(filename + ".obj");

    int outer_it = 0;
    Eigen::MatrixXd meshV = meshVoxelArap->meshV_;
    Eigen::MatrixXd meshV1 = meshVoxelArap->meshV_;
    Eigen::VectorXi b;
    meshVoxelArap->precompute_arap_data(b);

    std::cout << "Start Optimization..." << std::endl;

    double obj_tot = 1E-6;
    while (outer_it < 1)
    {
        meshVoxelArap->meshV_ = meshV1;
        meshVoxelArap->voxelization(Vs, Fs, volumes, areas, voxel_indices);
        meshVoxelArap->computeSelectedVoxels(volumes, voxel_indices);
        meshVoxelArap->meshV_ = meshV;

        int num_iters = 20;
        // Set up parameters
        LBFGSpp::LBFGSParam<double> param;
        param.epsilon = 1e-8;
        param.max_iterations = 100;

        while (num_iters--) {
            meshVoxelArap->compute_rotation_matrices(meshV1);

            // Create solver and function object
            LBFGSpp::LBFGSSolver<double> solver(param);

            // Initial guess
            Eigen::VectorXd x;
            meshVoxelArap->flatten(meshV, x);

            double fx;
            int niter = solver.minimize(*meshVoxelArap, x, fx);

            meshVoxelArap->reshape(x, meshV1);

        }
//        while (num_iters--) {
//
//            std::cout << "Iteration:\t" << num_iters << std::endl;
//            meshVoxelArap->compute_rotation_matrices(meshV1);
//
//            std::cout << "Compute Rotation" << std::endl;
//
//            MeshVoxelARAP_Solver instance = MeshVoxelARAP_Solver(meshVoxelArap, meshV1);
//
//            std::cout << "Create Instance" << std::endl;
//
//            knitro::KNSolver solver(&instance);
//
//            /** Set option to print output after every iteration. */
//            //solver.setParam(KN_PARAM_OUTLEV, KN_OUTLEV_ITER);
//            solver.setParam(KN_PARAM_EVAL_FCGA, KN_EVAL_FCGA_YES);
//            solver.setParam(KN_PARAM_HESSOPT, KN_HESSOPT_LBFGS);
//            solver.setUserParams(meshVoxelArap.get());
//
//            std::cout << meshVoxelArap->meshV_.size() << std::endl;
//
//            solver.initProblem();
//            std::cout << "Init Problem.." << std::endl;
//
//
//            int solveStatus = solver.solve();
//
//            std::vector<double> x;
//            std::vector<double> lambda;
//            int nStatus = solver.getSolution(x, lambda);
//
//            Eigen::VectorXd x_vec(x.size());
//            for(int id = 0; id < x.size(); id ++){
//                x_vec(id) = x[id];
//            }
//
//            meshVoxelArap->reshape(x_vec, meshV1);
//        }
        igl::writeOBJ("../test1_" + std::to_string(outer_it) + ".obj", meshV1, meshVoxelArap->meshF_);
        outer_it++;
    }
//    meshVoxelArap->meshV_ = meshV1;
//    meshVoxelArap->voxelization(Vs, Fs, volumes, areas, voxel_indices);
//    std::filesystem::remove_all("../output");
//    std::filesystem::create_directory("../output");
//    for (int id = 0; id < Vs.size(); id++) {
//        Eigen::Vector3i index = voxel_indices[id];
//        std::string index_str = std::to_string(index[0]) +
//                                "_" + std::to_string(index[1]) +
//                                "_" + std::to_string(index[2]);
//
//        if (volumes[id] > meshVoxelArap->minimum_volume_) {
//            igl::writeOBJ("../output/intersection_" + index_str + ".obj", Vs[id], Fs[id]);
//        } else {
//            igl::writeOBJ("../output/small" + index_str + ".obj", Vs[id], Fs[id]);
//        }
//    }
}
