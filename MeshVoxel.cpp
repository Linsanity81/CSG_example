//
// Created by ziqwang on 15.11.21.
//

#include "MeshVoxel.h"
#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <igl/fast_winding_number.h>
#include <fstream>
#include <map>

void MeshVoxel::readMesh(std::string filename) {
    igl::readOBJ(filename, meshV_, meshF_);
}

void MeshVoxel::voxelization_approximation_with_empty_voxels(vector<double> &volumes,
                                                  vector<Eigen::Vector3i> &voxel_indices){
    volumes.clear();
    voxel_indices.clear();

    int num_of_voxels = grids_size_ * grids_size_ * grids_size_;

    volumes.resize(num_of_voxels);
    voxel_indices.resize(num_of_voxels);

    int num_of_sample = 10;

    igl::FastWindingNumberBVH bvh;
    igl::fast_winding_number(meshV_, meshF_, 2, bvh);

    tbb::parallel_for( tbb::blocked_range<int>(0, num_of_voxels),
                       [&](tbb::blocked_range<int> r) {
        for (int id = r.begin(); id < r.end(); ++id)
        {
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            Eigen::Vector3i index = digit_to_index(id);

            Eigen::Vector3d corner(
                    index[0] * grids_width_ + grids_origin_[0],
                    index[1] * grids_width_ + grids_origin_[1],
                    index[2] * grids_width_ + grids_origin_[2]);

            Eigen::MatrixXd query_points((num_of_sample - 1) * (num_of_sample - 1) * (num_of_sample - 1), 3);
            int query_pt_index = 0;
            for(int ix = 1; ix < num_of_sample; ix++)
            {
                for(int iy = 1; iy < num_of_sample; iy++)
                {
                    for(int iz = 1; iz < num_of_sample; iz++)
                    {
                        Eigen::Vector3d pt =
                                corner + Eigen::Vector3d(ix * grids_width_ / num_of_sample,
                                                         iy * grids_width_ / num_of_sample,
                                                         iz * grids_width_ / num_of_sample);
                        query_points.row(query_pt_index) = pt;
                        query_pt_index++;
                    }
                }
            }

            Eigen::VectorXd winding;
            igl::fast_winding_number(bvh, 2, query_points, winding);

            double count = 0;
            for(int jd = 0; jd < winding.size(); jd++){
                count += winding(jd) > 0.5 ? 1: 0;
            }
            volumes[id] = count / query_pt_index * (grids_width_ * grids_width_ * grids_width_);
            voxel_indices[id] = index;
        }
    });
}

void MeshVoxel::voxelization_approximation(vector<double> &volumes,
                                           vector<Eigen::Vector3i> &voxel_indices){

    voxelization_approximation_with_empty_voxels(volumes, voxel_indices);

//    double min_volume = 1E-6;
    double min_volume = 0;

    vector<double> volumes_tmp;
    vector<Eigen::Vector3i> voxel_indices_tmp;

    for(int id = 0; id < volumes.size(); id++){
        if(volumes[id] > min_volume){
            volumes_tmp.push_back(volumes[id]);
            voxel_indices_tmp.push_back(voxel_indices[id]);
        }
    }
    volumes = volumes_tmp;
    voxel_indices = voxel_indices_tmp;

    return;
}

void MeshVoxel::compute_voxel(Eigen::Vector3i index, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    V = Eigen::MatrixXd::Zero(8, 3);
    F = Eigen::MatrixXi::Zero(12, 3);

    V << 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 1.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 1.0,
      1.0, 1.0, 0.0,
      1.0, 1.0, 1.0;

    F << 0, 6, 4,
    0, 2, 6,
    0, 3, 2,
    0, 1, 3,
    2, 7, 6,
    2, 3, 7,
    4, 6, 7,
    4, 7, 5,
    0, 4, 5,
    0, 5, 1,
    1, 5, 7,
    1, 7, 3;

    for(int id = 0; id < V.rows(); id++){
        V.row(id) *= grids_width_;
        V.row(id) += grids_origin_.transpose();
        Eigen::Vector3d offset = index.cast<double>();
        offset *= grids_width_;
        V.row(id) += offset.transpose();
    }

    return;
}

void MeshVoxel::write_voxels(std::string filename)
{
    std::ofstream fout(filename);

    fout << grids_origin_[0] << " " << grids_origin_[1] << " " << grids_origin_[2] << " " << grids_width_ << std::endl;
    fout << grids_size_ << std::endl;

    vector<double> volumes;
    vector<Eigen::Vector3i> voxel_indices;
    voxelization_approximation_with_empty_voxels(volumes, voxel_indices);

    for(int id = 0; id < volumes.size(); id++){
        fout << volumes[id] / grids_width_ / grids_width_ / grids_width_ << " ";
    }

    fout << std::endl;
    fout.close();
}

void MeshVoxel::write_voxels_full_volume(std::string filename)
{
    std::ofstream fout(filename);

    fout << grids_origin_[0] << " " << grids_origin_[1] << " " << grids_origin_[2] << " " << grids_width_ << std::endl;
    fout << grids_size_ << std::endl;

    vector<double> volumes;
    vector<Eigen::Vector3i> voxel_indices;
//    voxelization_approximation_with_empty_voxels(volumes, voxel_indices);
    voxelization_approximation(volumes, voxel_indices);
//    computeSelectedVoxels(volumes, voxel_indices, 0);

    for(int id = 0; id < volumes.size(); id++){
        fout << volumes[id] / grids_width_ / grids_width_ / grids_width_ << " ";
    }

    fout << std::endl;
    fout.close();
}

double MeshVoxel::computeDistanceVoxelToVoxel(Eigen::Vector3i voxelA, Eigen::Vector3i voxelB) const{
    Eigen::Vector3i distance = voxelA - voxelB;
    double minimum_distance = 0;
    for(int kd = 0; kd < 3; kd++){
        double d = std::abs(distance[kd]);
        minimum_distance += d > 0 ? pow((d - 1) * grids_width_, 2): 0;
    }
    return minimum_distance;
}

void MeshVoxel::computeDiffDistancePointToVoxel(Eigen::Vector3d pt,
                                                   Eigen::Vector3i voxel_index,
                                                   double &distance,
                                                   Eigen::Vector3d &gradient) const{
    distance = 0;
    gradient = Eigen::Vector3d(0, 0, 0);
    double eps = grids_width_ * 1E-2;
    for(int kd = 0; kd < 3; kd++)
    {
        double max_voxel_coord = grids_origin_[kd] + (voxel_index[kd] + 1) * grids_width_ - eps;
        double min_voxel_coord = grids_origin_[kd] + voxel_index[kd] * grids_width_ + eps;

        if(pt[kd] > max_voxel_coord)
        {
            distance += pow(pt[kd] - max_voxel_coord, 2.0);
            gradient[kd] = 2 * (pt[kd] - max_voxel_coord);
            //distance += pt[kd] - max_voxel_coord;
            //gradient[kd] = 1;
        }
        else if(pt[kd] < min_voxel_coord){
            distance += pow(min_voxel_coord - pt[kd], 2.0);
            gradient[kd] = 2 * (pt[kd] - min_voxel_coord);
            //distance += min_voxel_coord - pt[kd];
            //gradient[kd] = -1;
        }
    }
}

void MeshVoxel::computeSelectedVoxels(vector<double> &volumes, vector<Eigen::Vector3i> &voxel_indices, double ratio)
{
    double minimum_volume = ratio * grids_width_ * grids_width_ * grids_width_;
    selected_voxel_indices_.clear();
    for(int id = 0; id < volumes.size(); id++){
        if(volumes[id] > minimum_volume){
            selected_voxel_indices_.push_back(voxel_indices[id]);
        }
    }
    return;
}

int MeshVoxel::computePartialFullnTinyVoxels(vector<double> &volumes, double ratio){
    int count = 0;
    double minimum_volume = ratio * grids_width_ * grids_width_ * grids_width_;
    for(int id = 0; id < volumes.size(); id++){
        if(volumes[id] > minimum_volume){
            count++;
        }
        else if(volumes[id] < 0.05 * grids_width_ * grids_width_ * grids_width_){
            count ++;
        }
    }
    return count;
}

void MeshVoxel::expansion_voxels(const vector<Eigen::Vector3i> &input_voxels,
                                 vector<Eigen::Vector3i> &expension_voxels){
    expension_voxels.clear();

    auto comp = [](const Eigen::Vector3i &a, const Eigen::Vector3i &b){
        if(a[0] < b[0])
        {
            return true;
        }
        else if(a[0] == b[0] && a[1] < b[1]){
            return true;
        }
        else if(a[0] == b[0] && a[1] == b[1] && a[2] < b[2]){
            return true;
        }
        return false;
    };

    std::map<Eigen::Vector3i, bool, decltype(comp)> voxel_visited(comp);

    for(int id = 0; id < input_voxels.size(); id++){
        voxel_visited[input_voxels[id]] = true;
    }

    for(int id = 0; id < input_voxels.size(); id++)
    {
        Eigen::Vector3i index = input_voxels[id];
        for(int ix = -1; ix <= 1; ix++)
        {
            for(int iy = -1; iy <= 1; iy ++)
            {
                for(int iz = -1; iz <= 1; iz++)
                {
//                    if(abs(ix) + abs(iy) + abs(iz) <= 1){
                        Eigen::Vector3i new_index = index + Eigen::Vector3i(ix, iy, iz);
                        if(voxel_visited.find(new_index) == voxel_visited.end()){
                            expension_voxels.push_back(new_index);
                            voxel_visited[new_index] = true;
//                        }
                    }
                }
            }
        }
    }

}

void MeshVoxel::cluster_points_to_voxel_groups(const Eigen::MatrixXd &tv,
                                               vector<vector<int>> &group_pts,
                                               vector<Eigen::Vector3i> &group_voxel_indices)
                                               const{

    //1) compute which voxel the given point belongs to
    vector<std::pair<int, Eigen::Vector3i>> pts_voxel;
    for(int id = 0; id < tv.rows(); id++){
        Eigen::Vector3i voxel_index = point_to_voxel_index(tv.row(id));
        std::pair<int, Eigen::Vector3i> data;
        data.first = id;
        data.second = voxel_index;
        pts_voxel.push_back(data);
    }

    //2) sort the points according the voxel they belong to
    std::sort(pts_voxel.begin(), pts_voxel.end(), [](std::pair<int, Eigen::Vector3i>a, std::pair<int, Eigen::Vector3i> b){
        if(a.second[0] < b.second[0])
        {
            return true;
        }
        else if(a.second[0] == b.second[0] && a.second[1] < b.second[1]){
            return true;
        }
        else if(a.second[0] == b.second[0] && a.second[1] == b.second[1] && a.second[2] < b.second[2]){
            return true;
        }
        return false;
    });

    //3) cluster the points that are in the same voxel
    for(int id = 0; id < pts_voxel.size(); id++){
        if(group_voxel_indices.empty() ||
        group_voxel_indices.back() != pts_voxel[id].second){
            group_voxel_indices.push_back(pts_voxel[id].second);
            group_pts.push_back(vector<int>());
            group_pts.back().push_back(pts_voxel[id].first);
        }
        else{
            group_pts.back().push_back(pts_voxel[id].first);
        }
    }
}

void MeshVoxel::sort_input_voxels_respect_to_distance_to_given_voxel_group(Eigen::Vector3i voxel_group_index,
                                                                          const vector<Eigen::Vector3i> &input_voxels,
                                                                          vector<Eigen::Vector3i> &output_voxels,
                                                                          vector<double> &distance) const
                                                                          {
    vector<std::pair<int, double>> datas;

    for(int id = 0; id < input_voxels.size(); id++){
        Eigen::Vector3i input_voxel_index = input_voxels[id];
        double distance = computeDistanceVoxelToVoxel(voxel_group_index, input_voxel_index);
        datas.push_back({id, distance});
    }

    std::sort(datas.begin(), datas.end(), [](std::pair<int, double> a, std::pair<int, double>b){
        return a.second < b.second;
    });

    for(int id = 0; id < datas.size(); id++){
        output_voxels.push_back(input_voxels[datas[id].first]);
        distance.push_back(datas[id].second);
    }

    return;
}

Eigen::Vector3i MeshVoxel::point_to_voxel_index(Eigen::Vector3d pt) const{
    int nx = std::floor((pt[0] - grids_origin_[0]) / grids_width_);
    int ny = std::floor((pt[1] - grids_origin_[1]) / grids_width_);
    int nz = std::floor((pt[2] - grids_origin_[2]) / grids_width_);
    return Eigen::Vector3i(nx, ny, nz);
}

void MeshVoxel::subdivide_triangle(int faceID,
                                   const Eigen::MatrixXd& meshV1,
                                   Eigen::MatrixXd curr_tri_bary_coords,
                                   vector<Eigen::Vector3d> &bary_coords,
                                   vector<int> &bary_coords_faceID) const{
    Eigen::MatrixXd curr_tri = Eigen::MatrixXd::Zero(3, 3);
    for(int id = 0; id < 3; id++)
    {
        Eigen::Vector3d pt(0, 0, 0);
        for(int jd = 0; jd < 3; jd++)
        {
            int vID = meshF_(faceID, jd);
            pt += meshV1.row(vID) * curr_tri_bary_coords(id, jd);
        }
        curr_tri.row(id) = pt;
    }

    Eigen::Vector3d e1 = curr_tri.row(1) - curr_tri.row(0);
    Eigen::Vector3d e2 = curr_tri.row(2) - curr_tri.row(0);
    double area = (e1.cross(e2)).norm() * 0.5;
    if(area < 1E-4){
        return;
    }

    Eigen::Vector3d center_coord = (curr_tri_bary_coords.row(0) + curr_tri_bary_coords.row(1) + curr_tri_bary_coords.row(2)) / 3;
    int indices[3][2] = {
            {0, 1},
            {1, 2},
            {2, 0}
    };

    bary_coords.push_back(center_coord);
    bary_coords_faceID.push_back(faceID);

    for(int id = 0; id < 3; id++){
        Eigen::MatrixXd next_tri_bary_coords(3, 3);
        next_tri_bary_coords.row(0) = center_coord;
        for(int jd = 0; jd < 2; jd++)
        {
            int index = indices[id][jd];
            next_tri_bary_coords.row(jd + 1) = curr_tri_bary_coords.row(index);
        }
        subdivide_triangle(faceID,
                           meshV1,
                           next_tri_bary_coords,
                           bary_coords,
                           bary_coords_faceID);
    }
}

void MeshVoxel::reshape(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat) const{
        mat = Eigen::MatrixXd(vec.size() / 3, 3);
        for(int id = 0; id < vec.rows() / 3; id++){
            mat(id, 0) = vec[3 * id];
            mat(id, 1) = vec[3 * id + 1];
            mat(id, 2) = vec[3 * id + 2];
        }
    }

void MeshVoxel::flatten(const Eigen::MatrixXd &mat, Eigen::VectorXd &vec) const {
    vec = Eigen::VectorXd::Zero(mat.rows() * 3);
    for(int id = 0; id < mat.rows(); id++) {
        vec[3 * id] = mat(id, 0);
        vec[3 * id + 1] = mat(id, 1);
        vec[3 * id + 2] = mat(id, 2);
    }
}

void MeshVoxel::compute_point_to_selected_voxels_distance(const Eigen::MatrixXd &tv,
                                                          double &distance,
                                                          Eigen::MatrixXd &gradient) const{

    vector<vector<int>> group_pts;
    vector<Eigen::Vector3i> group_voxel_indices;

    cluster_points_to_voxel_groups(tv, group_pts, group_voxel_indices);

    distance = 0;
    gradient = Eigen::MatrixXd::Zero(tv.rows(), 3);

    for(int id = 0; id < group_voxel_indices.size(); id++)
    {
        vector<Eigen::Vector3i> sorted_selected_voxels;
        vector<double> distances;
        sort_input_voxels_respect_to_distance_to_given_voxel_group(group_voxel_indices[id],
                                                                   selected_voxel_indices_,
                                                                   sorted_selected_voxels,
                                                                   distances);

        for(int iv = 0; iv < group_pts[id].size(); iv++)
        {
            int point_id = group_pts[id][iv];
            Eigen::Vector3d pt = tv.row(point_id);
            double point_distance = std::numeric_limits<double>::max();
            Eigen::Vector3d point_gradient;

            for(int jd = 0; jd < sorted_selected_voxels.size(); jd++)
            {
                if(point_distance < distances[jd]){
                    break;
                }

                Eigen::Vector3i selected_voxel_index = sorted_selected_voxels[jd];

                double curr_point_voxel_distance;
                Eigen::Vector3d curr_point_voxel_distance_graident;
                computeDiffDistancePointToVoxel(pt,
                                                selected_voxel_index,
                                                curr_point_voxel_distance,
                                                curr_point_voxel_distance_graident);

                if(curr_point_voxel_distance < point_distance){
                    point_distance = curr_point_voxel_distance;
                    point_gradient = curr_point_voxel_distance_graident;
                }
            }

            distance += point_distance;
            gradient.row(point_id) = point_gradient;
        }
    }

    return;
}

void MeshVoxel::compute_triangle_to_selected_voxels_distance(const Eigen::MatrixXd &meshV1,
                                                             double &distance,
                                                             Eigen::MatrixXd &gradient) const {
    vector<Eigen::Vector3d> bary_coords;
    vector<int> bary_coords_faceID;
    for(int id = 0; id < meshF_.rows(); id++){
        Eigen::MatrixXd bary_coord(3, 3);

        bary_coord << 1, 0, 0,
                0, 1, 0,
                0, 0, 1;

        subdivide_triangle(id,
                           meshV1,
                           bary_coord,
                           bary_coords,
                           bary_coords_faceID);
    }

    Eigen::MatrixXd tv(bary_coords.size(), 3);
    for(int id = 0;id < bary_coords.size(); id++)
    {
        Eigen::Vector3d pt(0, 0, 0);
        for(int jd = 0; jd < 3; jd++)
        {
            int fID = bary_coords_faceID[id];
            int vID = meshF_(fID, jd);
            pt += meshV1.row(vID) * bary_coords[id][jd];
        }
        tv.row(id) = pt;
    }

    distance = 0;
    Eigen::MatrixXd grad_wrt_bary;
    compute_point_to_selected_voxels_distance(tv, distance, grad_wrt_bary);

    gradient = Eigen::MatrixXd::Zero(meshV1.rows(), 3);

    for(int id = 0; id < bary_coords.size(); id++)
    {
        for(int jd = 0; jd < 3; jd++)
        {
            int fID = bary_coords_faceID[id];
            int vID = meshF_(fID, jd);
            gradient.row(vID) += grad_wrt_bary.row(id) * bary_coords[id][jd];
        }
    }
}


void MeshVoxel::compute_point_to_voxels_distance(const Eigen::MatrixXd &tv,
                                                 const vector<Eigen::Vector3i> &boundary_voxels,
                                                 const vector<Eigen::Vector3i> &core_voxels,
                                                 double tolerance,
                                                 double &distance,
                                                 Eigen::MatrixXd &gradient) const{

    vector<vector<int>> group_pts;
    vector<Eigen::Vector3i> group_voxel_indices;
    cluster_points_to_voxel_groups(tv, group_pts, group_voxel_indices);

    double gap = tolerance * grids_width_;

    auto comp = [](const Eigen::Vector3i &a, const Eigen::Vector3i &b){
        if(a[0] < b[0])
        {
            return true;
        }
        else if(a[0] == b[0] && a[1] < b[1]){
            return true;
        }
        else if(a[0] == b[0] && a[1] == b[1] && a[2] < b[2]){
            return true;
        }
        return false;
    };

    std::map<Eigen::Vector3i, bool, decltype(comp)> map_boundary_voxels(comp), map_core_voxels(comp);

    vector<Eigen::Vector3i> all_voxels;
    for(int id = 0; id < boundary_voxels.size(); id++){
        map_boundary_voxels[boundary_voxels[id]] = true;
        all_voxels.push_back(boundary_voxels[id]);
    }

    for(int id = 0; id < core_voxels.size(); id++){
        map_core_voxels[core_voxels[id]] = true;
        all_voxels.push_back(core_voxels[id]);
    }

    gradient = Eigen::MatrixXd::Zero(tv.rows(), 3);
    distance = 0;

    for(int id = 0; id < group_voxel_indices.size(); id++)
    {
        vector<Eigen::Vector3i> sorted_voxels;
        vector<double> distances;
        int group_voxel_type = -1;

        if(map_boundary_voxels.find(group_voxel_indices[id]) != map_boundary_voxels.end()){
            //boundary
            sort_input_voxels_respect_to_distance_to_given_voxel_group(group_voxel_indices[id],
                                                                       core_voxels,
                                                                       sorted_voxels,
                                                                       distances);
            group_voxel_type = 0;
        }
        else if(map_core_voxels.find(group_voxel_indices[id]) != map_core_voxels.end()){
            //core
            sort_input_voxels_respect_to_distance_to_given_voxel_group(group_voxel_indices[id],
                                                                       boundary_voxels,
                                                                       sorted_voxels,
                                                                       distances);
            group_voxel_type = 1;
        }
        else{
            //outer space
            sort_input_voxels_respect_to_distance_to_given_voxel_group(group_voxel_indices[id],
                                                                       all_voxels,
                                                                       sorted_voxels,
                                                                       distances);
            group_voxel_type = 2;
        }

        for(int iv = 0; iv < group_pts[id].size(); iv++)
        {
            int point_id = group_pts[id][iv];
            Eigen::Vector3d pt = tv.row(point_id);
            double point_distance = std::numeric_limits<double>::max();
            Eigen::Vector3d point_gradient;
            Eigen::Vector3i closest_voxel;

            for(int jd = 0; jd < sorted_voxels.size(); jd++)
            {
                if(point_distance < distances[jd]){
                    break;
                }

                Eigen::Vector3i selected_voxel_index = sorted_voxels[jd];

                double curr_point_voxel_distance;
                Eigen::Vector3d curr_point_voxel_distance_graident;
                computeDiffDistancePointToVoxel(pt,
                                                selected_voxel_index,
                                                curr_point_voxel_distance,
                                                curr_point_voxel_distance_graident);

                if(curr_point_voxel_distance < point_distance){
                    point_distance = curr_point_voxel_distance;
                    point_gradient = curr_point_voxel_distance_graident;
                    closest_voxel = selected_voxel_index;
                }
            }

            if(group_voxel_type == 0){
                //boundary
                if(point_distance < gap * gap){
                    //gap * gap - point_distance
                    distance += gap * gap - point_distance;
                    gradient.row(point_id) = -point_gradient;
                }
            }
            else if(group_voxel_type == 1){
                distance += gap * gap + point_distance;
                gradient.row(point_id) = point_gradient;

//                std::cout << group_voxel_indices[id].transpose() << std::endl;
//                std::cout << pt.transpose() << std::endl;
//                std::cout << closest_voxel.transpose() << std::endl;
//                std::cout << point_distance << std::endl;
//                std::cout << point_gradient.transpose() << std::endl;
//                std::cout << std::endl;
            }
            else{
                distance += point_distance;
                gradient.row(point_id) = point_gradient;
            }
        }
    }
}

void MeshVoxel::compute_triangle_to_voxels_distance(const Eigen::MatrixXd &meshV1,
                                                    const vector<Eigen::Vector3i> &boundary_voxels,
                                                    const vector<Eigen::Vector3i> &core_voxels,
                                                    double tolerance,
                                                    double &distance,
                                                    Eigen::MatrixXd &gradient) const{
    vector<Eigen::Vector3d> bary_coords;
    vector<int> bary_coords_faceID;
    for(int id = 0; id < meshF_.rows(); id++){
        Eigen::MatrixXd bary_coord(3, 3);

        bary_coord << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;

        subdivide_triangle(id,
                           meshV1,
                           bary_coord,
                           bary_coords,
                           bary_coords_faceID);
    }

    Eigen::MatrixXd tv(bary_coords.size(), 3);
    for(int id = 0;id < bary_coords.size(); id++)
    {
        Eigen::Vector3d pt(0, 0, 0);
        for(int jd = 0; jd < 3; jd++)
        {
            int fID = bary_coords_faceID[id];
            int vID = meshF_(fID, jd);
            pt += meshV1.row(vID) * bary_coords[id][jd];
        }
        tv.row(id) = pt;
    }

    distance = 0;
    Eigen::MatrixXd grad_wrt_bary;
    compute_point_to_voxels_distance(tv, boundary_voxels, core_voxels, tolerance, distance, grad_wrt_bary);

    gradient = Eigen::MatrixXd::Zero(meshV1.rows(), 3);

    for(int id = 0; id < bary_coords.size(); id++)
    {
        for(int jd = 0; jd < 3; jd++)
        {
            int fID = bary_coords_faceID[id];
            int vID = meshF_(fID, jd);
            gradient.row(vID) += grad_wrt_bary.row(id) * bary_coords[id][jd];
        }
    }
}