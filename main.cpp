#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include "MeshVoxel.h"
int main(int argc, char *argv[])
{
    std::string mesh_obj_str = "../data/Bunny_12x12x9.obj";
    std::string puzzle_piece_str = "../data/grid1_piece1.obj";

    MeshVoxel mesh(Eigen::Vector3d(-1, -1, -1), 0.4);
    mesh.readMesh(mesh_obj_str);
    mesh.voxelization(5);
    return 0;
}
