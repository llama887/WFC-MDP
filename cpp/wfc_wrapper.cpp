#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include "wfc.cpp"

namespace nb = nanobind;

NB_MODULE(wfc_cpp, m) {
    // // Custom propagator python initialization
    // m.def("create_propagator_state", [](const nb::list &py_rules){
    //     Propagator::PropagatorState cpp_rules;
        
    //     for (size_t i = 0; i < nb::len(py_rules); i++) {
    //         std::array<std::vector<unsigned>, 4> pattern_rules;
    //         nb::list pattern_list = py_rules[i].cast<nb::list>();

    //         for (size_t j = 0; j < 4 && j < nb::len(pattern_list); ++j) {
    //             nb::list direction_list = pattern_list[j].cast<nb::list>();

    //             for (size_t k = 0; k < nb::len(direction_list); k++) {
    //                 pattern_rules[j].push_back(direction_list[k].cast<unsigned>());
    //             }
    //         }
    //         cpp_rules.push_back(pattern_rules);
    //     }
    //     return cpp_rules;
    // });

    nb::class_<WFC>(m, "WFC")
        .def(nb::init<bool, int, std::vector<double>, 
            Propagator::PropagatorState, unsigned, unsigned>())
        .def("get_wave_state", &WFC::get_wave_state)
        .def("get_next_collapse_cell", &WFC::get_next_collapse_cell)
        .def("collapse_step", &WFC::collapse_step)
        .def("run", &WFC::run);
        
    // nb::class_<Array3D<bool>>(m, "Array3D")
    //     .def("get", &Array3D<bool>::get)
    //     .def("set", &Array3D<bool>::set)
    //     .def_ro("height", &Array3D<bool>::height)
    //     .def_ro("width", &Array3D<bool>::width) 
    //     .def_ro("depth", &Array3D<bool>::depth);
}