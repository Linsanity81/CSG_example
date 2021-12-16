//
// Created by ziqwang on 16.12.21.
//

#ifndef MESHVOXEL_MESHVOXELARAP_SOLVER_H
#define MESHVOXEL_MESHVOXELARAP_SOLVER_H

#include "KNSolver.h"
#include "KNProblem.h"
#include "MeshVoxelARAP.h"

class MeshVoxelARAP_Solver : public knitro::KNProblem {
public:

    /*------------------------------------------------------------------*/
    /*     FUNCTION callbackEvalFC                                      */
    /*------------------------------------------------------------------*/
    static int callbackEvalFCGA(KN_context_ptr kc,
                              CB_context_ptr cb,
                              KN_eval_request_ptr const evalRequest,
                              KN_eval_result_ptr const evalResult,
                              void *const userParams) {
        const double *x;
        double *obj;
        double *c;
        double *objGrad;

        if (evalRequest->type != KN_RC_EVALFCGA)
        {
            printf ("*** callbackEvalFC incorrectly called with eval type %d\n",
                    evalRequest->type);
            return( -1 );
        }

        x = evalRequest->x;
        objGrad = evalResult->objGrad;

        Eigen::MatrixXd meshV1;
        const MeshVoxelARAP* mesh = (const MeshVoxelARAP *)userParams;

        Eigen::VectorXd x_vec(mesh->meshV_.size());
        for(int id = 0; id < x_vec.size(); id++){
            x_vec(id) = x[id];
        }

        Eigen::VectorXd grad;
        *obj = (*mesh).operator()(x_vec, grad);

        for(int id = 0; id < x_vec.size(); id++){
            objGrad[id] = grad[id];
        }

        return 0;
    }

    MeshVoxelARAP_Solver(std::shared_ptr<MeshVoxelARAP> mesh,
                         Eigen::MatrixXd &meshV1)
    : KNProblem(mesh->meshV_.size(),0) {
        Eigen::VectorXd x;
        mesh->flatten(meshV1, x);
        vector<double> init_values(x.array().data(), x.array().data() + x.size());
        this->setXInitial(init_values);
        this->setObjEvalCallback(&MeshVoxelARAP_Solver::callbackEvalFCGA);
    }
};


#endif //MESHVOXEL_MESHVOXELARAP_SOLVER_H
