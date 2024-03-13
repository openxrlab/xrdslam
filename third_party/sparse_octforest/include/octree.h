#ifndef _OCTREE_H
#define _OCTREE_H

#include <memory>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>


//                _____
//              /____ /|
//             |     | |
//     (0,0,0) |_____|/
//              side
// voxel coordinated defined on (0,0,0)
// side indicate the voxel length at current level
// point inside voxel would cast to int at (0,0,0)

class Octant : public torch::CustomClassHolder
{
public:
    inline Octant(int octree_index, int octant_index)
    {
        octree_index_ = octree_index;
        octant_index_ = octant_index;
        code_ = 0;
        side_ = 0;
        depth_ = -1;
        is_leaf_ = false;
        point_cnt_ = 0; // record how many points inside voxel
        child_ptr_ = std::vector<std::shared_ptr<Octant>>(8, nullptr);
        // std::cout << "[Debug][Octant] create octant in octree: " << octree_index_ << ", with index: " << octant_index_ <<std::endl;
    }
    ~Octant() {}
    std::shared_ptr<Octant>& child(const int offset)
    {
        return child_ptr_[offset];
    }

    int octree_index_;
    int octant_index_;
    uint64_t code_;
    unsigned int side_;
    int point_cnt_;
    int depth_;
    bool is_leaf_;
    std::vector<std::shared_ptr<Octant>> child_ptr_;
};


class Octree : public torch::CustomClassHolder
{
public:
    ~Octree();
    Octree(int64_t max_voxels);
    void insert(torch::Tensor points);
    double try_insert(torch::Tensor pts);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> get_all();
    bool has_voxels(torch::Tensor points);
    torch::Tensor get_features(torch::Tensor points);


    int size_;
    // temporal solution for serialization
    std::vector<torch::Tensor> all_pts_;
    Octree(int64_t grid_dim, std::vector<torch::Tensor> all_pts);

private:
    std::pair<int64_t, int64_t> count_nodes_internal();
    std::pair<int64_t, int64_t> count_recursive_internal(std::shared_ptr<Octant> n);
    std::shared_ptr<Octant> find_octant(std::vector<int> coord);
    std::shared_ptr<Octant> find_octant(int x, int y, int z);

    int octree_idx_;
    int octant_cnt_;
    static int octree_cnt_;
    std::shared_ptr<Octant> root_;
    std::set<uint64_t> all_codes;

    int max_level_;
};

torch::Tensor get_features_cuda(torch::Tensor points,
                                torch::Tensor children);

#endif
