#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/edges.h>
#include <vector>
#include "MeshVoxel.h"
#include <igl/writeOBJ.h>
#include "SurfaceEvo.h"
#include "SurfaceSlice.h"
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include "igl/file_dialog_save.h"
using std::vector;

void add_edges(igl::opengl::glfw::Viewer& viewer, std::shared_ptr<MeshVoxel> meshVoxel)
{
    for(int id = 0; id < meshVoxel->selected_voxel_indices_.size(); id++)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        meshVoxel->compute_voxel(meshVoxel->selected_voxel_indices_[id], V, F);

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

void generate_evolution_surface(Eigen::MatrixXd &V, Eigen::MatrixXi &F){

    //Vase1
//    vector<double> data_xs = {0,
//                              0.3,
//                              0.4,
//                              0.6,
//                              1.1};
//
//    vector<double> data_yts = {0.4, -1,
//                               0.3, 0.3,
//                               0.45, 0.3,
//                               0.45, -0.3,
//                               0.3, -0.3};


    //Vase 2
//    vector<double> data_xs = {0,
//                              0.8,
//                              1.2};
//
//    vector<double> data_yts = {0.2, 0.1,
//                               0.5, 1,
//                               0.3, -1};


    //Mush Room
//    vector<double> data_xs = {0,
//                              0.5,
//                              0.6,
//                              1.0};
//
//    vector<double> data_yts = {0.25, 1,
//                               0.5, 0,
//                               0.2, -1,
//                               0.35, 1};

    //Table
        vector<double> data_xs = {0,
                              0.2,
                              0.3,
                              0.7,
                              0.8,
                              1.0};

    vector<double> data_yts = {0.5, 0,
                               0.5, 0,
                               0.2, 0.0,
                               0.2, 0.0,
                               0.5, 0.0,
                               0.5, 0.0};

    SurfaceEvo surface(data_xs);
    surface.computeMesh(data_yts, V, F);

    return;
}

void optimize_mesh(std::shared_ptr<MeshVoxel> meshVoxel,
                   Eigen::MatrixXd &V,
                   Eigen::MatrixXi &F,
                   double smooth_weight)
{
    double grids_size = meshVoxel->grids_size_;
    double grids_width = meshVoxel->grids_width_;
    Eigen::Vector3d grids_origin = meshVoxel->grids_origin_;


    vector<double> volumes;
    vector<Eigen::Vector3i> voxel_indices;

    meshVoxel->voxelization_approximation(volumes, voxel_indices);
    meshVoxel->computeSelectedVoxels(volumes, voxel_indices);

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    for(int id = 0; id < meshVoxel->selected_voxel_indices_.size(); id++){
        double x = meshVoxel->selected_voxel_indices_[id].x() * grids_width + grids_origin[0];
        min_x = std::min(x, min_x);
        max_x = std::max(x + grids_width, max_x);
    }

    SurfaceSlice surfaceSlice(min_x, max_x);
    surfaceSlice.initSurface(grids_origin,
                             grids_width,
                             meshVoxel->selected_voxel_indices_);

    vector<vector<double>> new_radius;
    surfaceSlice.optimize(smooth_weight, new_radius);

    surfaceSlice.radius_ = new_radius;
    surfaceSlice.computeMesh(V, F);

    meshVoxel->meshV_ = V;
    meshVoxel->meshF_ = F;
    meshVoxel->voxelization_approximation(volumes, voxel_indices);
    meshVoxel->computeSelectedVoxels(volumes, voxel_indices);
    std::cout << "Full Voxel Percentage:\t" << meshVoxel->selected_voxel_indices_.size() / (double) volumes.size() << std::endl;
}

int main()
{
    std::shared_ptr<MeshVoxel> meshVoxel;

    int grids_size = 15;
    double grids_width = 0.08;
    double voxel_ratio = 0.3;
    double shape_weight = 10.0;

    Eigen::MatrixXd input_V;
    Eigen::MatrixXi input_F;

    Eigen::MatrixXd drawing_V;
    Eigen::MatrixXi drawing_F;

    igl::opengl::glfw::Viewer viewer;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    double doubleVariable;
    // Add content to the default menu window
    menu.callback_draw_custom_window = [&]() {
        ImGui::Begin(
                "Menu", nullptr,
                ImGuiWindowFlags_AlwaysAutoResize
                );

        // Add new group
        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Expose variable directly ...
            ImGui::InputInt("Grids Size", &grids_size, 1, 1);
            ImGui::InputDouble("Grids Width", &grids_width, 0.01, 0.02, "%.2f");
            ImGui::InputDouble("Full Voxel Ratio", &voxel_ratio, 0.025, 0.1, "%.2f");
            ImGui::InputDouble("Shape Weight", &shape_weight, 1, 10, "%.2f");

        }

        if (ImGui::CollapsingHeader("Inputs", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Expose variable directly ...
            if (ImGui::Button("Generate Evolution Surface", ImVec2(-1,0)))
            {
                generate_evolution_surface(input_V, input_F);
                drawing_V = input_V;
                drawing_F = input_F;
                viewer.data().clear();
                viewer.core().align_camera_center(drawing_V, drawing_F);
                viewer.data().set_mesh(drawing_V, drawing_F);
            }
        }

        if (ImGui::CollapsingHeader("Actions", ImGuiTreeNodeFlags_DefaultOpen)) {

            if (ImGui::Button("Voxelization", ImVec2(-1,0)))
            {
                Eigen::Vector3d origin(0, 0, 0);
                origin[1] = -grids_width * grids_size / 2;
                origin[2] = -grids_width * grids_size / 2;

                meshVoxel = std::make_shared<MeshVoxel>(origin, grids_width, grids_size, voxel_ratio);
                meshVoxel->meshV_ = input_V;
                meshVoxel->meshF_ = input_F;
                drawing_F = input_F;
                drawing_V = input_V;
                vector<double> volumes;
                vector<Eigen::Vector3i> voxel_indices;
                meshVoxel->voxelization_approximation(volumes, voxel_indices);
                meshVoxel->computeSelectedVoxels(volumes, voxel_indices);
                viewer.data().clear();
                viewer.data().set_mesh(drawing_V, drawing_F);
                viewer.core().align_camera_center(drawing_V, drawing_F);
                add_edges(viewer, meshVoxel);

                std::cout << "Full Voxel Percentage:\t" << meshVoxel->selected_voxel_indices_.size() / (double) volumes.size() << std::endl;
            }
            if (ImGui::Button("Optimize Surface", ImVec2(-1,0)))
            {
                if(input_F.rows() != 0){
                    Eigen::Vector3d origin(0, 0, 0);
                    origin[1] = -grids_width * grids_size / 2;
                    origin[2] = -grids_width * grids_size / 2;

                    meshVoxel = std::make_shared<MeshVoxel>(origin, grids_width, grids_size, voxel_ratio);
                    meshVoxel->meshV_ = input_V;
                    meshVoxel->meshF_ = input_F;
                    optimize_mesh(meshVoxel, drawing_V, drawing_F, shape_weight);
                    viewer.data().clear();
                    viewer.data().set_mesh(drawing_V, drawing_F);
                    viewer.core().align_camera_center(drawing_V, drawing_F);
                    add_edges(viewer, meshVoxel);
                }
            }

            if(ImGui::Button("Save Surface", ImVec2(-1,0))){
                std::string filename = igl::file_dialog_save();
                igl::writeOBJ(filename + ".obj", drawing_V, drawing_F);
                meshVoxel->write_voxels(filename + ".puz");
            }
        }

        ImGui::End();
    };

    viewer.launch();
}