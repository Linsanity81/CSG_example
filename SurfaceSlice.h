//
// Created by ziqwang on 29.12.21.
//

#ifndef MESHVOXEL_SURFACESLICE_H
#define MESHVOXEL_SURFACESLICE_H

#include <vector>
#include <Eigen/Dense>
#include "KNSolver.h"
#include "KNProblem.h"
using namespace knitro;
using std::vector;

using std::vector;
class SurfaceSliceOpt : public knitro::KNProblem {
public:

    SurfaceSliceOpt(const vector<double> &xs,
                  const vector<vector<double>> &radius,
                  int nvar,
                  int ncon,
                  double weight) : KNProblem(nvar + ncon, ncon){

        // Variables init
        {
            vector<double> var_init_value;
            vector<int> var_init_index;
            int iv = 0;
            for(int id = 0; id < radius.size(); id++){
                for(int jd = 0; jd < radius[id].size(); jd++){
                    var_init_value.push_back(radius[id][jd]);
                    var_init_index.push_back(iv);
                    iv++;
                }
            }
            this->setXInitial({var_init_index, var_init_value});
        }

        // Variables bounds
        {
            vector<int> var_bnds_indices;
            vector<double> var_lo_bnds_value;
            vector<double> var_up_bnds_values;

            int iv = 0;
            for(int id = 0; id < radius.size(); id++){
                for(int jd = 0; jd < radius[id].size(); jd++){
                    var_bnds_indices.push_back(iv);
                    var_lo_bnds_value.push_back(0.0);
                    var_up_bnds_values.push_back(radius[id][jd]);
                    iv ++;
                }
            }
            this->setVarLoBnds({var_bnds_indices, var_lo_bnds_value});
            this->setVarUpBnds({var_bnds_indices, var_up_bnds_values});
        }

        // Constraints Equal
        {
            vector<double> conEqBnds;
            vector<int> conIndices;
            for(int id = 0; id < ncon; id++){
                conIndices.push_back(id);
                conEqBnds.push_back(0);
            }
            this->setConEqBnds({conIndices, conEqBnds});
        }

        //set Equality
        {
            vector<int> con_indices;
            vector<vector<int>> var_indices;
            vector<vector<double>> var_coeffs;

            int NS = radius.size();
            int NT = radius.front().size();
            int icon = 0;
            for(int id = 0; id < NS; id++)
            {
                for(int jd = 0; jd < NT; jd++)
                {
                    int prev_iv = id * NT + (jd - 1 + NT) % NT;
                    int next_iv = id * NT + (jd + 1) % NT;
                    int iv = id * NT + jd;
                    con_indices.push_back(icon);
                    var_indices.push_back({prev_iv, next_iv, iv, iv + nvar});
                    var_coeffs.push_back({1.0, 1.0, -2.0, -1.0});
                    icon++;
                }
            }

            int iz = 2 * nvar;
            for(int id = 1; id + 1 < NS; id++)
            {
                for(int jd = 0; jd < NT; jd++)
                {
                    int prev_iv = (id - 1) * NT + jd;
                    int next_iv = (id + 1) * NT + jd;
                    int iv = id * NT + jd;
                    con_indices.push_back(icon);
                    var_indices.push_back({prev_iv, next_iv, iv, iz});
                    var_coeffs.push_back({1.0, 1.0, -2.0, -1.0});
                    icon++;
                    iz++;
                }
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
            int iv = 0;
            for(int id = 0; id < radius.size(); id++){
                for(int jd = 0; jd < radius[id].size(); jd++){
                    obj_index.push_back(iv);
                    obj_quad_coeff.push_back(0.5);
                    obj_linar_coeff.push_back(-radius[id][jd]);
                    iv++;
                }
            }
            for(int id = nvar; id < nvar + ncon; id++){
                obj_index.push_back(id);
                obj_quad_coeff.push_back(weight);
                obj_linar_coeff.push_back(0.0);
            }

            this->setObjectiveQuadraticPart({obj_index, obj_index, obj_quad_coeff});
            this->setObjectiveLinearPart({obj_index, obj_linar_coeff});
        }
    }
};

class SurfaceSlice{
public:

    vector<vector<double>> radius_;

    vector<double> xs_;

    int num_theta_sample_;

    int num_x_sample_;
public:

    SurfaceSlice(double x0, double x1){
        num_theta_sample_ = 128;
        num_x_sample_ = 100;
        for(int id = 1; id <= num_x_sample_; id++){
            xs_.push_back((double) id / (num_x_sample_ + 1) * (x1 - x0) + x0);
        }
    }

    void initSurface(Eigen::Vector3d grids_origin,
                     double grids_width,
                     int grids_size,
                     const vector<Eigen::Vector3i> &voxel_indices);

    void compute_voxel(Eigen::Vector3i index,
                       Eigen::Vector3d grids_origin,
                       double grids_width,
                       Eigen::MatrixXd &V,
                       Eigen::MatrixXi &F);

    void computeMesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F);

};


#endif //MESHVOXEL_SURFACESLICE_H
