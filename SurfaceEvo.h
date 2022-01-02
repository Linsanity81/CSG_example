//
// Created by ziqwang on 28.12.21.
//

#ifndef MESHVOXEL_SURFACEEVO_H
#define MESHVOXEL_SURFACEEVO_H

#include <vector>
#include <Eigen/Dense>
#include "KNSolver.h"
#include "KNProblem.h"
using namespace knitro;

using std::vector;
class SurfaceShrink : public knitro::KNProblem {
public:

    SurfaceShrink(const vector<double> &data_xs,
              const vector<double> &yts,
              const vector<double> &xs,
              const Eigen::MatrixXd &conMat,
              const Eigen::VectorXd &conB,
              double weight) : KNProblem(yts.size() + xs.size(),
                                         xs.size()){

        // Variables init
        {
            vector<double> var_init_value;
            for(int id = 0; id < yts.size(); id++){
                var_init_value.push_back(0.0);
            }
            for(int id = 0; id < xs.size(); id++){
                var_init_value.push_back(0.0);
            }
            this->setXInitial(var_init_value);
        }

        // Variables bounds
        {
            vector<int> var_lo_bnds_indices;
            vector<double> var_lo_bnds_value;
            for(int id = yts.size(); id < xs.size() + yts.size(); id++){
                var_lo_bnds_indices.push_back(id);
                var_lo_bnds_value.push_back(0.0);
            }
            this->setVarLoBnds({var_lo_bnds_indices, var_lo_bnds_value});

            vector<double> var_up_bnds_values;
            for(int id = 0; id < xs.size(); id++){
                var_up_bnds_values.push_back(conB[id]);
            }
            this->setVarUpBnds({var_lo_bnds_indices, var_up_bnds_values});
        }


        // Constraints Equal
        {
            vector<double> conLoBnds;
            vector<int> conIndices;
            for(int id = 0; id < xs.size(); id++){
                conIndices.push_back(id);
                conLoBnds.push_back(0);
            }
            this->setConEqBnds({conIndices, conLoBnds});
        }

        /** Add linear structure and coefficients. */
        {
            vector<int> con_indices;
            vector<vector<int>> var_indices;
            vector<vector<double>> var_coeffs;

            var_indices.resize(conMat.rows());
            var_coeffs.resize(conMat.rows());

            for(int id = 0; id < conMat.rows(); id++)
            {
                con_indices.push_back(id);
                for(int jd = 0; jd < conMat.cols(); jd++)
                {
                    var_indices[id].push_back(jd);
                    var_coeffs[id].push_back(conMat(id, jd));
                }
                var_indices[id].push_back(id + yts.size());
                var_coeffs[id].push_back(-1);
            }

            KNSparseVector<int,KNLinearStructure> conLinPart;
            for(int id = 0; id < con_indices.size(); id++)
            {
                KNLinearStructure structure(var_indices[id], var_coeffs[id]);
                conLinPart.add(con_indices[id], structure);
            }
            this->setConstraintsLinearParts(conLinPart);
        }

        /** Set the coefficients for the objective. */
        {
            vector<int> obj_index;
            vector<double> obj_quad_coeff;
            vector<double> obj_linar_coeff;

            for(int id = 0; id < xs.size(); id++)
            {
                obj_index.push_back(yts.size() + id);
                obj_quad_coeff.push_back(0.5);
                obj_linar_coeff.push_back(-conB[id]);
            }

            this->setObjectiveQuadraticPart({obj_index, obj_index, obj_quad_coeff});
            this->setObjectiveLinearPart({obj_index, obj_linar_coeff});
        }
    }
};


class SurfaceEvo {

public:

    vector<double> data_xs_;

public:

    SurfaceEvo(const vector<double> &data_xs){
        data_xs_ = data_xs;
    }

    void compute_ys(const vector<double> &data_yts,
                    const Eigen::VectorXd &xs,
                    Eigen::VectorXd &ys);

    Eigen::MatrixXd computeB(double x0, double x1);

    void divide_xs(const Eigen::VectorXd &xs,
                   vector<vector<double>> &group_xs);


    void computeMesh(const vector<double> &data_yts, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

    void computeRadius(Eigen::Vector3d grids_origin,
                       double grids_width,
                       int grids_size,
                       const vector<Eigen::Vector3i> &selected_voxel_indices,
                       const vector<double> &xs,
                       vector<double> &radius);

    void compute_constraints(const vector<double> &xs,
                             vector<double> &radius,
                             Eigen::MatrixXd &Mat,
                             Eigen::VectorXd &b);


};


#endif //MESHVOXEL_SURFACEEVO_H
