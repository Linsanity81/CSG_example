#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOBJ.h>
#include <igl/barycenter.h>
#include <igl/edges.h>
#include <vector>
#include "MeshVoxel.h"
#include "MeshVoxelOpt.h"
#include "LBFGS.h"
#include <igl/writeOBJ.h>
#include "knitro.h"
#include "KNSolver.h"
#include "KNProblem.h"

using std::vector;
using namespace knitro;

vector<Eigen::MatrixXd> Vs;

vector<Eigen::MatrixXi> Fs;

vector<double> volumes;

vector<vector<double>> areas;

vector<Eigen::Vector3i> voxel_indices;

Eigen::Vector3d grids_origin;

double grids_width;

int grids_size;

std::shared_ptr<MeshVoxelOpt> meshVoxelOpt;


void add_edges(igl::opengl::glfw::Viewer& viewer){
    for(int id = 0; id < meshVoxelOpt->selected_voxel_indices.size(); id++){
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        meshVoxelOpt->compute_voxel(meshVoxelOpt->selected_voxel_indices[id], V, F);

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

class ProblemNLP2 : public knitro::KNProblem {
public:

    /*------------------------------------------------------------------*/
    /*     FUNCTION callbackFCGA                                       */
    /*------------------------------------------------------------------*/
    static int callbackFCGA(KN_context_ptr             kc,
                             CB_context_ptr             cb,
                             KN_eval_request_ptr const  evalRequest,
                             KN_eval_result_ptr  const  evalResult,
                             void              * const  userParams){

        if (evalRequest->type != KN_RC_EVALFCGA)
        {
            printf ("*** callbackEvalH incorrectly called with eval type %d\n",
                    evalRequest->type);
            return( -1 );
        }

        MeshVoxelOpt *mesh = (MeshVoxelOpt *)userParams;

        int num_of_vars = mesh->TV_.size();
        Eigen::VectorXd x(num_of_vars);
        for(int id = 0; id < num_of_vars; id++){
            x(id) = evalRequest->x[id];
        }

        double obj;
        Eigen::VectorXd gradient;
        obj = mesh->operator()(x, gradient);

        *evalResult->obj = obj;
        for(int id = 0; id < num_of_vars; id++){
            evalResult->objGrad[id] = gradient[id];
        }

        return( 0 );
    }

    void setSolver(KNSolver * solver) {
        this->getNewPointCallback().setParams(solver);
    }


    ProblemNLP2(int num_of_vars) : KNProblem(num_of_vars,0)
    {
        /** Plug the callback "callbackEvalFCGA" */
        this->setObjEvalCallback(&ProblemNLP2::callbackFCGA);

        /** Set minimize or maximize (if not set, assumed minimize) */
        this->setObjGoal(KN_OBJGOAL_MINIMIZE);
    }
};

int main() {

    grids_origin = Eigen::Vector3d(-1, -1, -1);
    grids_size = 10;
    grids_width = 2.0 / grids_size;

    meshVoxelOpt = std::make_shared<MeshVoxelOpt>(grids_origin, grids_width, grids_size, 0.3);
    meshVoxelOpt->readMesh("../data/Bunny_12x12x9.obj", "pa0.0001q1.414Y");
    //meshVoxelOpt->approxVoxelization(Vs, Fs, volumes, voxel_indices);
    meshVoxelOpt->voxelization(Vs, Fs, volumes, areas, voxel_indices);
    meshVoxelOpt->computeSelectedVoxels(volumes, voxel_indices);

//    double distance;
//    Eigen::VectorXd gradient;
//    Eigen::VectorXd tv_x;
//    meshVoxelOpt->flatten(meshVoxelOpt->TV_, tv_x);
//    distance = meshVoxelOpt->operator()(tv_x, gradient);
//    std::cout << distance << std::endl;

    // Create a problem instance.
    ProblemNLP2 instance = ProblemNLP2(meshVoxelOpt->TV_.size());
    Eigen::VectorXd init_x;
    meshVoxelOpt->flatten(meshVoxelOpt->TV_, init_x);
    vector<double> init_x_vec(init_x.array().data(), init_x.array().data() + init_x.rows());

    std::cout << "(1)" << std::endl;
    instance.setXInitial(init_x_vec);

    std::cout << "(2)" << std::endl;

    // Create a solver
    knitro::KNSolver solver(&instance);
    instance.setSolver(&solver);

    std::cout << "(3)" << std::endl;

    /** Set option to print output after every iteration. */
    solver.setParam(KN_PARAM_OUTLEV, 0);
    solver.setParam(KN_PARAM_EVAL_FCGA, KN_EVAL_FCGA_YES);
    solver.setParam(KN_PARAM_HESSOPT, KN_HESSOPT_LBFGS);

    std::cout << "(4)" << std::endl;

    solver.initProblem();
    solver.setUserParams(meshVoxelOpt.get());

    std::cout << "(5)" << std::endl;
    solver.solve();

    std::cout << "(6)" << std::endl;

    std::vector<double> x;
    std::vector<double> lambda;
    int nStatus = solver.getSolution(x, lambda);

    Eigen::VectorXd result_x(x.size());
    for(int id = 0; id < x.size(); id++){
        result_x[id] = x[id];
    }

    Eigen::MatrixXd tv;
    meshVoxelOpt->reshape(result_x, tv);

    Eigen::MatrixXi F = meshVoxelOpt->TF_;
    for(int id = 0; id < F.rows(); id++){
        std::swap(F(id, 1), F(id, 2));
    }

    meshVoxelOpt->meshV_ = tv;
    meshVoxelOpt->meshF_ = F;
//    meshVoxelOpt->voxelization(Vs, Fs, volumes, areas, voxel_indices);

    // Plot the generated mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(tv, F);
    add_edges(viewer);

//    viewer.callback_key_down = &key_down;
//    key_down(viewer,'9',0);
    viewer.launch();

    return 0;
}

//int main(int argc, char *argv[])
//{
//    grids_origin = Eigen::Vector3d(-1, -1, -1);
//    grids_size = 6;
//    grids_width = 2.0 / grids_size;
//
//    MeshVoxelOpt meshVoxelOpt(grids_origin, grids_width, grids_size, 0.01);
//    meshVoxelOpt.readMesh("../data/Bunny_12x12x9.obj", "pa0.0001q1.41Y");
//    meshVoxelOpt.approxVoxelization(Vs, Fs, volumes, voxel_indices);
//    meshVoxelOpt.computeSelectedVoxels(volumes);
//
//    LBFGSpp::LBFGSParam<double> param;
//    param.epsilon = 1e-6;
//    param.max_iterations = 1000;
//    param.max_linesearch = 100;
//
//    // Create solver and function object
//    LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBracketing> solver(param);
//
//    // Initial guess
//
//    Eigen::VectorXd x;
//    Eigen::MatrixXd tv;
//    meshVoxelOpt.flatten(meshVoxelOpt.TV_, x);
//    // x will be overwritten to be the best point found
//    double fx;
//    int niter = solver.minimize(meshVoxelOpt, x, fx);
//
//    std::cout << niter << " iterations" << std::endl;
//    std::cout << "f(x) = " << fx << std::endl;
////
////    return 0;
//
//    meshVoxelOpt.reshape(x, tv);
//
//    //igl::writeOBJ("../bunny.obj", tv, meshVoxelOpt.TF_);
//
//    // Plot the generated mesh
//    igl::opengl::glfw::Viewer viewer;
//    Eigen::MatrixXi F = meshVoxelOpt.TF_;
//    for(int id = 0; id < F.rows(); id++){
//        std::swap(F(id, 1), F(id, 2));
//    }
//    viewer.data().set_mesh(tv, meshVoxelOpt.TF_);
////    viewer.callback_key_down = &key_down;
////    key_down(viewer,'9',0);
//    viewer.launch();
//}