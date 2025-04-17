#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include "wfc.cpp"

namespace nb = nanobind;

NB_MODULE(wfc_cpp, m) {
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