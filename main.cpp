#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/edges.h>
#include <vector>
#include "MeshVoxel.h"
#include "MeshVoxelARAP.h"
#include <igl/writeOBJ.h>
#include "SurfaceEvo.h"
#ifdef __linux__
//    #include <filesystem>
//    namespace fs = std::filesystem;
#else
    #include <filesystem>
    namespace fs = std::__fs::filesystem;
#endif
#include "MeshVoxelARAP_Solver.h"

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

int main(){

    vector<double> data_xs = {0, 0.25, 0.5, 0.75, 1,};

    vector<double> data_yts = {0.3, 0, 0.4, 0, 0.5, 0, 0.4, 0, 0.3, 0};

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    SurfaceEvo surface(data_xs);


    double grids_size = 15;
    double grids_width = 0.1;
    Eigen::Vector3d grids_origin = Eigen::Vector3d( 0,-grids_width * grids_size / 2, -grids_width * grids_size / 2);

    int times = 5;
    while(times --)
    {
        surface.computeMesh(data_yts, V, F);
        meshVoxelArap = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
        meshVoxelArap->meshV_ = V;
        meshVoxelArap->meshF_ = F;
        meshVoxelArap->voxelization_approximation(volumes, voxel_indices);
        meshVoxelArap->computeSelectedVoxels(volumes, voxel_indices);
        std::cout << meshVoxelArap->selected_voxel_indices_.size() / (double)volumes.size() << std::endl;

        vector<double> xs;
        int num_sample = 100;
        for(int id = 0; id < num_sample; id++){
            xs.push_back(1.0 / num_sample * id);
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

        // Create a problem instance.
        SurfaceShrink instance = SurfaceShrink(data_xs, data_yts, xs, Mat, b, 1.0);

        // Create a solver
        knitro::KNSolver solver = knitro::KNSolver(&instance);
        solver.setParam(KN_PARAM_OUTLEV, 0);

        solver.initProblem();
        int solveStatus = solver.solve();

        std::vector<double> lambda;
        std::vector<double> x;

        int nStatus = solver.getSolution(x, lambda);
        for(int id = 0; id < data_yts.size(); id++){
            data_yts[id] = x[id];
        }
//        for(int id = data_yts.size(); id < x.size(); id++){
//            std::cout << radius[id - data_yts.size()] << " " << x[id] << std::endl;
//        }
    }

    surface.computeMesh(data_yts, V, F);
//    meshVoxelArap = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
//    meshVoxelArap->meshV_ = V;
//    meshVoxelArap->meshF_ = F;
//    meshVoxelArap->voxelization_approximation(volumes, voxel_indices);
//    meshVoxelArap->computeSelectedVoxels(volumes, voxel_indices);
//    std::cout << meshVoxelArap->selected_voxel_indices_.size() / (double)volumes.size() << std::endl;

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    add_edges(viewer);
    viewer.launch();
}


//int main() {
//    //Eigen::Vector3d grids_origin = Eigen::Vector3d(-1.1, -1, -1.2);
//    //Eigen::Vector3d grids_origin = Eigen::Vector3d( -1, -1.2, -1.05);
//    Eigen::Vector3d grids_origin = Eigen::Vector3d( -1,-1, -1);
//    //double grids_size = 8;
//    double grids_size = 9;
//    double grids_width = 0.25 ;
//
//    meshVoxelArap = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.3);
//    std::string filename = "../data/Model/Organic/Teddy";
//    meshVoxelArap->readMesh(filename + ".obj");
//
//    Eigen::MatrixXd meshV1 = meshVoxelArap->meshV_;
//    std::shared_ptr<MeshVoxelARAP_Solver> solver = std::make_shared<MeshVoxelARAP_Solver>(meshVoxelArap);
//
//    Eigen::Vector3d opt_grids_origin;
//    solver->optimize(meshV1, opt_grids_origin);
//
//    igl::writeOBJ("../Duck_22_12_21/Teddy.obj", meshV1, meshVoxelArap->meshF_);
//    meshVoxelArap->meshV_ = meshV1;
//    meshVoxelArap->grids_origin_ = opt_grids_origin;
//    meshVoxelArap->voxelization_approximation(volumes, voxel_indices);
//    meshVoxelArap->computeSelectedVoxels(volumes, voxel_indices);
//    std::cout << (double)meshVoxelArap->selected_voxel_indices_.size() / volumes.size() << std::endl;
//    meshVoxelArap->write_voxels("../Duck_22_12_21/Teddy.puz");
//
//
//////
//    meshVoxelArap->voxelization(Vs, Fs, volumes, areas, voxel_indices);
////    fs::remove_all("../output");
////    fs::create_directory("../output");
////    for (int id = 0; id < Vs.size(); id++) {
////        Eigen::Vector3i index = voxel_indices[id];
////        std::string index_str = std::to_string(index[0]) +
////                                "_" + std::to_string(index[1]) +
////                                "_" + std::to_string(index[2]);
////
////        if (volumes[id] > meshVoxelArap->minimum_volume_) {
////            igl::writeOBJ("../output/intersection_" + index_str + ".obj", Vs[id], Fs[id]);
////        } else {
////            igl::writeOBJ("../output/small" + index_str + ".obj", Vs[id], Fs[id]);
////        }
////    }
//}
