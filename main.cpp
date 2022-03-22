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

void add_voxels(igl::opengl::glfw::Viewer& viewer,
                std::shared_ptr<MeshVoxel> meshVoxel,
                const std::vector<Eigen::Vector3i> &voxel_indices
                )
{
    for(int id = 0; id < voxel_indices.size(); id++)
    {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        meshVoxel->compute_voxel(voxel_indices[id], V, F);

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

    vector<Eigen::MatrixXd> intermediate_results_;

    int grids_size = 10;
    double grids_width = 0.25;
    float grids_origin[3]
    = {-1, -1, -1};
    double full_voxel_ratio = 0.3;
    double core_voxel_ratio = 0.9;


    double gap = 0.5;
    double shape_weight = 200.0;
    double shape_weight_last_iteration = 200.0;

    bool location_optimization = false;

    int num_location_sample = 5;
    int num_outer_iterations = 1;
    int num_inner_iterations = 10;

    double lbfgs_eps = 1E-8;
    int lbfgs_iterations = 30;
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
        meshVoxel->computeSelectedVoxels(volumes, voxel_indices, full_voxel_ratio);
        viewer.data().clear();
        viewer.data().set_mesh(drawing_V, drawing_F);
        viewer.core().align_camera_center(drawing_V, drawing_F);
        add_voxels(viewer, meshVoxel, meshVoxel->selected_voxel_indices_);
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
            ImGui::InputDouble("Full Voxel Ratio", &full_voxel_ratio, 0.025, 0.1, "%.2f");
            ImGui::InputDouble("Core Voxel Ratio", &core_voxel_ratio, 0.025, 0.1, "%.2f");

        }

        if(ImGui::CollapsingHeader("Opt Para", ImGuiTreeNodeFlags_DefaultOpen)){
            ImGui::Checkbox("Location Optimization", &location_optimization);

            ImGui::InputDouble("Gap", &gap);
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

                meshVoxel = std::make_shared<MeshVoxelARAP>(origin, grids_width, grids_size);
                meshVoxel->meshV_ = input_V;
                meshVoxel->meshF_ = input_F;
                drawing_F = input_F;
                drawing_V = input_V;
                voxelization();
            }

            if (ImGui::Button("Expension", ImVec2(-1,0)))
            {
                if(meshVoxel){
                    vector<Eigen::Vector3i> expension_voxels;
                    meshVoxel->expansion_voxels(meshVoxel->selected_voxel_indices_, expension_voxels);
                    viewer.data().clear();
                    drawing_F = meshVoxel->meshF_;
                    drawing_V = meshVoxel->meshV_;
                    viewer.data().set_mesh(drawing_V, drawing_F);
                    add_voxels(viewer, meshVoxel, expension_voxels);
                }
            }

            if (ImGui::Button("Optimize Location", ImVec2(-1,0)))
            {
                if(input_F.rows() != 0){
                    Eigen::Vector3d origin(grids_origin[0], grids_origin[1], grids_origin[2]);
                    meshVoxel = std::make_shared<MeshVoxelARAP>(origin, grids_width, grids_size);
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

            if (ImGui::Button("Optimize", ImVec2(-1,0)))
            {
                if(input_F.rows() != 0){
                    Eigen::Vector3d origin(grids_origin[0], grids_origin[1], grids_origin[2]);
                    meshVoxel = std::make_shared<MeshVoxelARAP>(origin, grids_width, grids_size);
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
                    solver->gap_ = gap;

                    solver->use_location_optimization_ = location_optimization;

                    solver->full_voxel_ratio_ = full_voxel_ratio;
                    solver->core_voxel_ratio_ = core_voxel_ratio;

                    Eigen::MatrixXd meshV1 = input_V;

                    Eigen::Vector3d opt_grids_origin;
                    solver->optimize(meshV1, opt_grids_origin);

                    drawing_F = input_F;
                    drawing_V = meshV1;

                    meshVoxel->grids_origin_ = opt_grids_origin;
                    meshVoxel->meshF_ = drawing_F = input_F;
                    meshVoxel->meshV_ = drawing_V = meshV1;

                    intermediate_results_ = solver->intermediate_results_;

                    voxelization();
                }
            }

            if(ImGui::Button("Save Surface", ImVec2(-1,0))){
                std::string filename = igl::file_dialog_save();
                igl::writeOBJ(filename + ".obj", drawing_V, drawing_F);
                meshVoxel->write_voxels(filename + ".puz");

                for (int i = 0; i < intermediate_results_.size(); i += 10)
                {
                    std::string intermediateFileName = filename + "_" + std::to_string(i);
                    igl::writeOBJ(intermediateFileName + ".obj", intermediate_results_[i], drawing_F);
                }
//                meshVoxel->write_voxels_full_volume(filename + "_full_volume.puz");
            }


        }

        ImGui::End();
    };

    viewer.launch();
}
