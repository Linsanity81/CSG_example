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
    Eigen::Vector3d grids_origin = Eigen::Vector3d(-1, -1, -1);
    double grids_size = 10;
    double grids_width = 2.0 / grids_size;

    meshVoxelArap = std::make_shared<MeshVoxelARAP>(grids_origin, grids_width, grids_size, 0.5);
    //std::string filename = "../data/Bunny_12x12x9";
    std::string filename = "../data/Model/Organic/Duck";
    meshVoxelArap->readMesh(filename + ".obj");
//    meshVoxelArap->voxelization(Vs, Fs, volumes, areas, voxel_indices);
//    meshVoxelArap->computeSelectedVoxels(volumes, voxel_indices);
//    std::ofstream fout(filename + "_voxel.txt");
//    fout << meshVoxelArap->selected_voxel_indices_.size() << std::endl;
//    for(int id = 0; id < meshVoxelArap->selected_voxel_indices_.size(); id++){
//        for(int jd = 0; jd < 3; jd++){
//            fout << meshVoxelArap->selected_voxel_indices_[id][jd] << " ";
//        }
//        fout << std::endl;
//    }
//    fout.close();

    std::ifstream fin(filename + "_voxel.txt");
    int num_voxels;
    fin >> num_voxels;
    for(int id = 0; id < num_voxels; id++)
    {
        int x, y ,z;
        fin >> x >> y >> z;
        meshVoxelArap->selected_voxel_indices_.push_back(Eigen::Vector3i(x, y, z));
    }

    int num_iters = 1;
    Eigen::MatrixXd meshV1 = meshVoxelArap->meshV_;
    vector<Eigen::MatrixXd> Rs;
    double learning_rate = 0.0001;
    while(num_iters --){
        meshVoxelArap->compute_rotation_matrices(meshV1, Rs);
        double E;
        Eigen::MatrixXd gradient;
        meshVoxelArap->compute_energy(meshV1, Rs, E, gradient);
        meshV1 = meshV1 - learning_rate * gradient;
        std::cout << E << std::endl;
    }

//    meshVoxelArap->meshV_ = meshV1;
//    meshVoxelArap->voxelization(Vs, Fs, volumes, areas, voxel_indices);

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(meshV1, meshVoxelArap->meshF_);
    add_edges(viewer);
//    viewer.callback_key_down = &key_down;
//    key_down(viewer,'9',0);
    viewer.launch();


}