#pragma once

#include <simsycl/system.hh>


namespace simsycl::test {

template<typename DeviceSetup>
void configure_device_with(DeviceSetup &&setup_device) {
    auto system = builtin_system;
    setup_device(system.devices.at("GPU"));
    configure_system(system);
}

}; // namespace simsycl::test
