#pragma once

#include <simsycl/sycl/vec.hh>
#include <simsycl/system.hh>

namespace simsycl::test {

template<typename DeviceSetup>
void configure_device_with(DeviceSetup &&setup_device) {
    auto system = builtin_system;
    setup_device(system.devices.at("GPU"));
    configure_system(system);
}

template<int Dimensions>
inline bool check_bool_vec(simsycl::sycl::vec<bool, Dimensions> a) {
    for(int i = 0; i < Dimensions; ++i) {
        if(!a[i]) { return false; }
    }
    return true;
}

}; // namespace simsycl::test
