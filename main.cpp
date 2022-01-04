#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/edges.h>
#include <vector>
#include "MeshVoxel.h"
#include "MeshVoxelARAP.h"
#include <igl/writeOBJ.h>
#include "igl/file_dialog_open.h"
#include "igl/file_dialog_save.h"
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include "MeshVoxelARAP_Solver.h"

using std::vector;

void add_edges(igl::opengl::glfw::Viewer& viewer,  std::shared_ptr<MeshVoxelARAP> meshVoxel)
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

int main()
{
    std::shared_ptr<MeshVoxelARAP> meshVoxel;

    int grids_size = 9;
    double grids_width = 0.25;
    float grids_origin[3]
    = {-1, -1, -1};
    double voxel_ratio = 0.3;


    double shape_weight = 20.0;
    double shape_weight_last_iteration = 0.1;

    int num_location_sample = 5;
    int num_outer_iterations = 2;
    int num_inner_iterations = 3;

    double lbfgs_eps = 1E-8;
    int lbfgs_iterations = 50;
    int lbfgs_linesearch_iterations = 100;

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

    auto voxelization = [&](){
        vector<double> volumes;
        vector<Eigen::Vector3i> voxel_indices;
        meshVoxel->voxelization_approximation(volumes, voxel_indices);
        meshVoxel->computeSelectedVoxels(volumes, voxel_indices);
        viewer.data().clear();
        viewer.data().set_mesh(drawing_V, drawing_F);
        viewer.core().align_camera_center(drawing_V, drawing_F);
        add_edges(viewer, meshVoxel);
        std::cout << "Full Voxel Percentage:\t" << meshVoxel->selected_voxel_indices_.size() / (double) volumes.size() << std::endl;
    };

    menu.callback_draw_custom_window = [&]() {
        ImGui::Begin(
                "Menu", nullptr,
                ImGuiWindowFlags_AlwaysAutoResize
        );

        // Add new group
        if (ImGui::CollapsingHeader("Grids Para", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Expose variable directly ...
            ImGui::InputInt("Grids Size", &grids_size, 1, 1);
            ImGui::InputDouble("Grids Width", &grids_width, 0.01, 0.02, "%.2f");
            ImGui::InputFloat3("Grids Origin", grids_origin);
            ImGui::InputDouble("Full Voxel Ratio", &voxel_ratio, 0.025, 0.1, "%.2f");
        }

        if(ImGui::CollapsingHeader("Opt Para", ImGuiTreeNodeFlags_DefaultOpen)){
            ImGui::InputDouble("Shape Weight", &shape_weight);
            ImGui::InputDouble("Shape Weight Last", &shape_weight_last_iteration);

            ImGui::InputDouble("LBFGS Epsilon", &lbfgs_eps);
            ImGui::InputInt("LBFGS Iterations", &lbfgs_iterations, 1, 1);
            ImGui::InputInt("LBFGS LineSearch", &lbfgs_linesearch_iterations, 1, 1);

            ImGui::InputInt("Opt Outer Iterations", &num_outer_iterations, 1, 1);
            ImGui::InputInt("Opt Inner Iterations", &num_inner_iterations, 1, 1);
            ImGui::InputInt("Location Sample", &num_location_sample, 1, 1);
        }

        if (ImGui::CollapsingHeader("Inputs", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Expose variable directly ...
            if (ImGui::Button("Read Mesh", ImVec2(-1,0)))
            {
                std::string filename = igl::file_dialog_open();
                igl::readOBJ(filename, input_V, input_F);
                drawing_V = input_V;
                drawing_F = input_F;
                viewer.data().clear();
                viewer.data().set_mesh(drawing_V, drawing_F);
                viewer.core().align_camera_center(drawing_V, drawing_F);
            }
        }

        if (ImGui::CollapsingHeader("Actions", ImGuiTreeNodeFlags_DefaultOpen)) {

            if (ImGui::Button("Voxelization", ImVec2(-1,0)))
            {
                Eigen::Vector3d origin(grids_origin[0], grids_origin[1], grids_origin[2]);

                meshVoxel = std::make_shared<MeshVoxelARAP>(origin, grids_width, grids_size, voxel_ratio);
                meshVoxel->meshV_ = input_V;
                meshVoxel->meshF_ = input_F;
                drawing_F = input_F;
                drawing_V = input_V;
                voxelization();
            }

            if (ImGui::Button("Optimize Location", ImVec2(-1,0)))
            {
                if(input_F.rows() != 0){
                    Eigen::Vector3d origin(grids_origin[0], grids_origin[1], grids_origin[2]);
                    meshVoxel = std::make_shared<MeshVoxelARAP>(origin, grids_width, grids_size, voxel_ratio);
                    meshVoxel->meshV_ = input_V;
                    meshVoxel->meshF_ = input_F;

                    std::shared_ptr<MeshVoxelARAP_Solver> solver
                    = std::make_shared<MeshVoxelARAP_Solver>(meshVoxel);

                    Eigen::Vector3d opt_grids_origin;
                    opt_grids_origin = solver->optimize_location(origin);

                    drawing_F = input_F;
                    drawing_V = input_V;

                    meshVoxel->grids_origin_ = opt_grids_origin;
                    voxelization();
                }
            }

            if (ImGui::Button("Optimize Mesh Location", ImVec2(-1,0)))
            {
                if(input_F.rows() != 0){
                    Eigen::Vector3d origin(grids_origin[0], grids_origin[1], grids_origin[2]);
                    meshVoxel = std::make_shared<MeshVoxelARAP>(origin, grids_width, grids_size, voxel_ratio);
                    meshVoxel->meshV_ = input_V;
                    meshVoxel->meshF_ = input_F;

                    std::shared_ptr<MeshVoxelARAP_Solver> solver
                            = std::make_shared<MeshVoxelARAP_Solver>(meshVoxel);

                    solver->lbfgs_eps_ = lbfgs_eps;
                    solver->lbfgs_linesearch_iterations_ = lbfgs_linesearch_iterations;
                    solver->lbfgs_iterations_ = lbfgs_iterations;

                    solver->max_inner_it_time_ = num_inner_iterations;
                    solver->max_outer_it_time_ = num_outer_iterations;
                    solver->num_location_sample_ = num_location_sample;

                    solver->shape_weight_last_iteration_ = shape_weight_last_iteration;
                    solver->shape_weight_ = shape_weight;

                    Eigen::MatrixXd meshV1 = input_V;

                    Eigen::Vector3d opt_grids_origin;
                    solver->optimize(meshV1, opt_grids_origin);

                    drawing_F = input_F;
                    drawing_V = meshV1;

                    meshVoxel->grids_origin_ = opt_grids_origin;
                    meshVoxel->meshF_ = drawing_F = input_F;
                    meshVoxel->meshV_ = drawing_V = meshV1;

                    voxelization();
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
