#include "simsycl/detail/system.hh"


namespace simsycl::detail {

std::vector<sycl::device> g_devices;
std::vector<sycl::platform> g_platforms;

void setup() {
    g_platforms.push_back(sycl::platform(platform_state{}));
    g_devices.push_back(sycl::device(device_state{g_platforms.back()}));
}

} // namespace simsycl::detail

namespace simsycl::sycl {

std::vector<device> device::get_devices(info::device_type /* TODO */) { return detail::g_devices; }
std::vector<platform> platform::get_platforms() { return detail::g_platforms; }

} // namespace simsycl::sycl
