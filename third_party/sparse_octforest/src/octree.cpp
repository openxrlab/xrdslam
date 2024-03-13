#include "octree.h"
#include "utils.h"
#include "cuda_utils.h"
#include <queue>
#include <iostream>
#include <bitset>

const int incr_x[8] = {0, 0, 0, 0, 1, 1, 1, 1};
const int incr_y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
const int incr_z[8] = {0, 1, 0, 1, 0, 1, 0, 1};

int Octree::octree_cnt_ = 0;

Octree::~Octree()
{
}

Octree::Octree(int64_t max_voxels)
{
    if (max_voxels < 1)
    {
        std::cout << "[Error][Octree] max voxels should greater than 0 !!!" << std::endl;
        return;
    }

    octree_idx_ = Octree::octree_cnt_++;
    octant_cnt_ = 0;

    size_ = max_voxels;
    max_level_ = log2(size_); // root level is 0
    if (max_level_ >= MAX_BITS)
    {
        std::cout << "[Error][Octree] max level should less than " << MAX_BITS-1 << " !!!" << std::endl;
        return;
    }

    root_ = std::make_shared<Octant>(octree_idx_, octant_cnt_++);
    root_->side_ = size_;
    // std::cout << "[Debug][Octree] create new octree: " << octree_idx_ << std::endl;
}

void Octree::insert(torch::Tensor pts)
{
    bool create_new_node = false;

    if (root_ == nullptr)
    {
        std::cout << "[Error][Octree] Octree not initialized !!!" << std::endl;
        return;
    }

    auto points = pts.accessor<int, 2>(); // (P, 3)
    if (points.size(1) != 3)
    {
        std::cout << "[Error][Octree] Point dimensions mismatch: inputs are " << points.size(1) << " expect 3 !!!" << std::endl;
        return;
    }

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            // compute morton code
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t code = encode(x, y, z);
            // std::cout << "[Debug][Octree] xyz: (" << x << ", " << y << ", " << z << "), morton code is: " << std::bitset<sizeof(uint64_t) * 8>(code) << std::endl;
            all_codes.insert(code);

            const unsigned int shift = MAX_BITS - max_level_ - 1;
            // std::cout << "[Debug][Octree] shift: " << shift << std::endl;

            auto n = root_;
            unsigned edge = size_ / 2;
            for (int d = 1; d <= max_level_; edge /= 2, ++d)
            {
                const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
                auto& tmp = n->child(childid);
                if (tmp == nullptr)
                {
                    const uint64_t t_code = code & MASK[d + shift];
                    // std::cout << "[Debug][Octree] morton code is: " << std::bitset<sizeof(uint64_t) * 8>(t_code) << " - " << octant_cnt_ << std::endl;
                    tmp = std::make_shared<Octant>(octree_idx_, octant_cnt_++);
                    tmp->code_ = t_code;
                    tmp->side_ = edge;
                    tmp->is_leaf_ = (d == max_level_);
                    // std::cout << "[Debug][Octree] create octant at level: " << d << ", with side: " << tmp->side_ << ", is leaf: " << tmp->is_leaf_ <<std::endl;
                    create_new_node = true;
                }
                if (j == 0)
                {
                    ++(tmp->point_cnt_);
                    // std::cout << "[Debug][Octree] child " << childid << " with points " << n->child(childid)->point_cnt_ << std::endl;
                }

                n = tmp;
            }
        }
    }

    // temporal solution for serialization
    if (create_new_node)
        all_pts_.push_back(pts);
}

double Octree::try_insert(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "Octree not initialized!" << std::endl;
    }

    auto points = pts.accessor<int, 2>();
    if (points.size(1) != 3)
    {
        std::cout << "Point dimensions mismatch: inputs are " << points.size(1) << " expect 3" << std::endl;
        return -1.0;
    }

    std::set<uint64_t> tmp_codes;

    for (int i = 0; i < points.size(0); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int x = points[i][0] + incr_x[j];
            int y = points[i][1] + incr_y[j];
            int z = points[i][2] + incr_z[j];
            uint64_t key = encode(x, y, z);

            tmp_codes.insert(key);
        }
    }

    std::set<int> result;
    std::set_intersection(all_codes.begin(), all_codes.end(),
                          tmp_codes.begin(), tmp_codes.end(),
                          std::inserter(result, result.end()));

    double overlap_ratio = 1.0 * result.size() / tmp_codes.size();
    return overlap_ratio;
}



std::shared_ptr<Octant> Octree::find_octant(int x, int y, int z)
{
    auto n = root_;
    unsigned edge = size_ / 2;
    for (int d = 1; d <= max_level_; edge /= 2, ++d)
    {
        const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        auto tmp = n->child(childid);
        if (tmp == nullptr)
        {
            std::cout << "[Error][Octree] voxel not found at " << x << ", " << y << ", " << z << " !!!" << std::endl;
            return nullptr;
        }
        n = tmp;
    }
    return n;
}

std::shared_ptr<Octant> Octree::find_octant(std::vector<int> coord)
{
    int x = coord[0];
    int y = coord[1];
    int z = coord[2];

    return find_octant(x, y, z);
}

std::pair<int64_t, int64_t> Octree::count_nodes_internal()
{
    return count_recursive_internal(root_);
}

std::pair<int64_t, int64_t> Octree::count_recursive_internal(std::shared_ptr<Octant> n)
{
    if (n == nullptr)
        return std::make_pair<int64_t, int64_t>(0, 0);

    if (n->is_leaf_)
        return std::make_pair<int64_t, int64_t>(1, 1);

    auto sum = std::make_pair<int64_t, int64_t>(1, 0);

    for (int i = 0; i < 8; i++)
    {
        auto temp = count_recursive_internal(n->child(i));
        sum.first += temp.first;
        sum.second += temp.second;
    }

    return sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> Octree::get_all()
{
    auto node_count = count_nodes_internal();
    auto total_count = node_count.first;
    auto leaf_count = node_count.second;
    // std::cout << "[Debug][Octree] count: " << node_count.first << " nodes with " << node_count.second << " leaves" <<std::endl;
    auto all_voxels = torch::zeros({total_count, 4}, dtype(torch::kInt32));
    auto all_children = -torch::ones({total_count, 8}, dtype(torch::kInt32));
    auto all_features = -torch::ones({total_count, 8}, dtype(torch::kInt32));

    std::queue<std::shared_ptr<Octant>> all_nodes;
    all_nodes.push(root_);

    while (!all_nodes.empty())
    {
        auto node_ptr = all_nodes.front();
        all_nodes.pop();

        auto xyz = decode(node_ptr->code_);
        std::vector<int> coords = {static_cast<int>(xyz[0]),
                                    static_cast<int>(xyz[1]),
                                    static_cast<int>(xyz[2]),
                                    static_cast<int>(node_ptr->side_)};
        auto voxel = torch::from_blob(coords.data(), {4}, dtype(torch::kInt32));
        all_voxels[node_ptr->octant_index_] = voxel;

        if (node_ptr->is_leaf_)
        {
            for (int i = 0; i < 8; ++i)
            {
                std::vector<int> vcoords = coords;
                vcoords[0] += incr_x[i];
                vcoords[1] += incr_y[i];
                vcoords[2] += incr_z[i];
                auto octant = find_octant(vcoords);
                all_features[node_ptr->octant_index_][i] = octant->octant_index_;
            }
        }

        for (int i = 0; i < 8; i++)
        {
            auto& child_ptr = node_ptr->child(i);
            if (child_ptr != nullptr)
            {
                all_children[node_ptr->octant_index_][i] = child_ptr->octant_index_;
                if (child_ptr->point_cnt_ > 0)
                {
                    // std::cout << "[Debug][Octree] child " << i << " with points " << child_ptr->point_cnt_ << std::endl;
                    all_nodes.push(child_ptr);
                }
            }
        }
    }
    // return std::make_tuple(all_voxels, all_children, all_features);
    return std::make_tuple(all_voxels, all_children, all_features, leaf_count); // return the number of leaf
}

// temporal solution for serialization
Octree::Octree(int64_t max_voxels, std::vector<torch::Tensor> all_pts)
{
    if (max_voxels < 1)
    {
        std::cout << "[Error][Octree] max voxels should greater than 0 !!!" << std::endl;
        return;
    }

    octree_idx_ = Octree::octree_cnt_++;
    octant_cnt_ = 0;

    size_ = max_voxels;
    max_level_ = log2(size_); // root level is 0
    if (max_level_ >= MAX_BITS)
    {
        std::cout << "[Error][Octree] max level should less than " << MAX_BITS-1 << " !!!" << std::endl;
        return;
    }

    root_ = std::make_shared<Octant>(octree_idx_, octant_cnt_++);
    root_->side_ = size_;
    std::cout << "[Debug][Octree] create new octree: " << octree_idx_ << std::endl;
    std::cout << "[Debug][Octree] serialization info: max_voxels:" << max_voxels << " vector<pts> size:" << all_pts.size() << std::endl;

    // for (auto &pt : all_pts_)
    for (auto &pt : all_pts)
    {
        insert(pt);
        std::cout << "[Debug][Octree] serialization insert points." << std::endl;
    }
}


torch::Tensor Octree::get_features(torch::Tensor pts)
{
    auto points = pts.accessor<int, 2>(); // (P, 3)
    int total_points = points.size(0);
    auto all_features = -torch::ones({total_points, 8}, dtype(torch::kInt32));

    if (root_ == nullptr)
    {
        std::cout << "[Error][Octree] Octree not initialized !!!" << std::endl;
        return all_features;
    }

    if (points.size(1) != 3)
    {
        std::cout << "[Error][Octree] Point dimensions mismatch: inputs are " << points.size(1) << " expect 3 !!!" << std::endl;
        return all_features;
    }

    for (int i = 0; i < total_points; ++i)
    {
        int x = points[i][0];
        int y = points[i][1];
        int z = points[i][2];

        auto n = root_;
        unsigned edge = size_ / 2;
        for (int d = 1; d <= max_level_; edge /= 2, ++d)
        {
            const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
            auto tmp = n->child(childid);
            if (tmp == nullptr)
            {
                std::cout << "[Error][Octree] voxel not found at " << x << ", " << y << ", " << z << " !!!" << std::endl;
                break;
            }
            n = tmp;
        }

        if (n != nullptr && n->is_leaf_ && n->point_cnt_ > 0) {
            for (int j = 0; j < 8; ++j)
            {
                int tx = x + incr_x[j];
                int ty = y + incr_y[j];
                int tz = z + incr_z[j];
                auto octant = find_octant(tx, ty, tz);
                all_features[i][j] = octant->octant_index_;
            }
        }
    }
    return all_features;
}


bool Octree::has_voxels(torch::Tensor pts)
{
    if (root_ == nullptr)
    {
        std::cout << "[Error][Octree] Octree not initialized !!!" << std::endl;
        return false;
    }

    auto points = pts.accessor<int, 2>(); // (P, 3)
    if (points.size(1) != 3)
    {
        std::cout << "[Error][Octree] Point dimensions mismatch: inputs are " << points.size(1) << " expect 3 !!!" << std::endl;
        return false;
    }

    int total_points = points.size(0);
    for (int i = 0; i < total_points; ++i)
    {
        int x = points[i][0];
        int y = points[i][1];
        int z = points[i][2];

        auto n = root_;
        unsigned edge = size_ / 2;
        for (int d = 1; d <= max_level_; edge /= 2, ++d)
        {
            const int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
            auto tmp = n->child(childid);
            if (tmp == nullptr)
            {
                std::cout << "[Error][Octree] voxel not found at " << x << ", " << y << ", " << z << " !!!" << std::endl;
                break;
            }
            n = tmp;
        }

        if (n != nullptr && n->is_leaf_ && n->point_cnt_ > 0) {
            return true;
        }
    }
    return false;
}

void get_features_kernel_wrapper(const int *points,
    const int *children,
    int *features,
    const int num_batch,
    const int num_sample);

torch::Tensor get_features_cuda(torch::Tensor points,
                                torch::Tensor children)
{

    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(children);
    CHECK_IS_INT(points);
    CHECK_IS_INT(children);
    CHECK_CUDA(points);
    CHECK_CUDA(children);

    torch::Tensor features = torch::full(
                        {points.size(0), points.size(1), 1},
                        -1,
                        torch::device(points.device()).dtype(torch::ScalarType::Int)
                        );
    get_features_kernel_wrapper(points.data_ptr<int>(),
                                children.data_ptr<int>(),
                                features.data_ptr<int>(),
                                points.size(0),
                                points.size(1));
    return features;
}
