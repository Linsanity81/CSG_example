#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
int main(int argc, char *argv[])
{
    std::string mesh_obj_str = "/Users/ziqwang/Documents/GitHub/CSG_example/Bunny_12x12x9_K12_Strict_1_T0.02/Bunny_12x12x9.obj";
    std::string puzzle_piece_str = "/Users/ziqwang/Documents/GitHub/CSG_example/Bunny_12x12x9_K12_Strict_1_T0.02/grid1_piece1.obj";

    //read OBJ
    Eigen::MatrixXd V0, V1, V2;
    Eigen::MatrixXi F0, F1, F2;

    igl::readOBJ(mesh_obj_str, V0, F0);
    igl::readOBJ(puzzle_piece_str, V1, F1);

    Eigen::VectorXi J;
    igl::copyleft::cgal::mesh_boolean(V0, F0, V1, F1, igl::MeshBooleanType::MESH_BOOLEAN_TYPE_INTERSECT, V2, F2, J);
    
    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V2, F2);
    viewer.data().set_face_based(true);
    viewer.launch();
}
