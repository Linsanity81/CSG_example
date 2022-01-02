#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/edges.h>
#include <vector>
#include "MeshVoxel.h"
#include <igl/writeOBJ.h>
#include "SurfaceEvo.h"
#include "SurfaceSlice.h"

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

int main()
{
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

    vector<vector<double>> new_radius;
    surfaceSlice.optimize(300, new_radius);

    surfaceSlice.radius_ = new_radius;
    surfaceSlice.computeMesh(V, F);

    meshVoxel = std::make_shared<MeshVoxel>(grids_origin, grids_width, grids_size, 0.3);
    meshVoxel->meshV_ = V;
    meshVoxel->meshF_ = F;
    meshVoxel->voxelization_approximation(volumes, voxel_indices);
    meshVoxel->computeSelectedVoxels(volumes, voxel_indices);
    std::cout << meshVoxel->selected_voxel_indices_.size() / (double)volumes.size() << std::endl;

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    add_edges(viewer, meshVoxel);
    viewer.launch();
}