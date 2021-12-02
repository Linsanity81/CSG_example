//#include <igl/opengl/glfw/Viewer.h>
//#include <igl/readOBJ.h>
//#include <igl/copyleft/cgal/mesh_boolean.h>
//#include "MeshVoxel.h"
//int main(int argc, char *argv[])
//{
//    std::string mesh_obj_str = "../data/Bunny_12x12x9.obj";
//    std::string puzzle_piece_str = "../data/grid1_piece1.obj";
//
//    MeshVoxel mesh(Eigen::Vector3d(-1, -1, -1), 0.4);
//    mesh.readMesh(mesh_obj_str);
//    mesh.voxelization(5);
//    return 0;
//}

#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/edges.h>
#include <vector>
#include "MeshVoxel.h"
#include "MeshVoxelOpt.h"

using std::vector;

vector<Eigen::MatrixXd> Vs;

vector<Eigen::MatrixXi> Fs;

vector<double> volumes;

vector<vector<double>> areas;

vector<Eigen::Vector3i> voxel_indices;

Eigen::Vector3d grids_origin;

double grids_width;

int grids_size;

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
    }


    return false;
}

int main(int argc, char *argv[])
{
    grids_origin = Eigen::Vector3d(-1, -1, -1);
    grids_size = 6;
    grids_width = 2.0 / grids_size;

//    MeshVoxel meshVoxel(grids_origin, grids_width, grids_size);
//
//    // Load a surface mesh
//    meshVoxel.readMesh("../data/Bunny_12x12x9.obj");
//    //voxelization
//    meshVoxel.voxelization(Vs, Fs, volumes, areas, voxel_indices);
//    for(int id = 0; id < volumes.size(); id++){
//        std::cout << voxel_indices[id].transpose() << ":\t" << volumes[id] << std::endl;
//    }

    MeshVoxelOpt meshVoxelOpt(grids_origin, grids_width, grids_size, 0.1);
    meshVoxelOpt.readMesh("../data/Bunny_12x12x9.obj", "pa0.0001q1.41Y");
    meshVoxelOpt.approxVoxelization(Vs, Fs, volumes, voxel_indices);
    vector<Eigen::Vector3i> selected_voxel_indices;
    meshVoxelOpt.computeSelectedVoxels(volumes, selected_voxel_indices);
    double distance;
    Eigen::MatrixXd gradient;
    meshVoxelOpt.computeDiffDistanceToSelectedVoxels(meshVoxelOpt.TV_,
                                                     selected_voxel_indices,
                                                     distance,
                                                     gradient);

    std::cout << meshVoxelOpt.TV_.rows() << std::endl;


    // Plot the generated mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = &key_down;
    key_down(viewer,'9',0);
    viewer.launch();
}
