#pragma once

#include "check.hh"

#include "../sycl/device.hh"
#include "../sycl/exception.hh"
#include "../sycl/platform.hh"

#include <vector>


namespace simsycl::detail {

extern std::vector<sycl::device> g_devices;
extern std::vector<sycl::platform> g_platforms;

void setup();

template<typename DeviceSelector>
sycl::device select_device(const DeviceSelector &selector) {
    SIMSYCL_CHECK(!g_devices.empty());
    int max_rating = INT_MIN;
    for(const auto &device : g_devices) {
        if(int rating = selector(device); rating > max_rating) { max_rating = rating; }
    }
    if(max_rating < 0) { throw sycl::exception(sycl::errc::runtime, "No suitable device found"); }
    const auto device = std::find_if(
        g_devices.begin(), g_devices.end(), [&](const auto &device) { return selector(device) == max_rating; });
    assert(device != g_devices.end());
    return *device;
}

} // namespace simsycl::detail
