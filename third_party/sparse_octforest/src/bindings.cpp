#include "octree.h"

TORCH_LIBRARY(forest, m)
{
    // m.def("encode", &encode_torch);
    m.class_<Octree>("Octree")
        .def(torch::init<int64_t>())
        .def("insert", &Octree::insert)
        .def("try_insert", &Octree::try_insert)
        .def("has_voxels", &Octree::has_voxels)
        .def("get_features", &Octree::get_features)
        .def("get_all", &Octree::get_all)
        .def_pickle(
        // __getstate__
        [](const c10::intrusive_ptr<Octree>& self) -> std::tuple<int64_t, std::vector<torch::Tensor>> {
            return std::make_tuple(self->size_, self->all_pts_);
        },
        // __setstate__
        [](std::tuple<int64_t, std::vector<torch::Tensor>> state) {
            return c10::make_intrusive<Octree>(std::get<0>(state), std::get<1>(state));
        })
        ;
    m.def("get_features_cuda", &get_features_cuda);
}
