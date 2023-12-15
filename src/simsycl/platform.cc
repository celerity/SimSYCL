#include "simsycl/sycl/platform.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/system.hh"

#include <algorithm>
#include <iterator>

namespace simsycl::detail {

struct platform_state {
    platform_config config;
    std::vector<sycl::device> devices;
};

} // namespace simsycl::detail

namespace simsycl::sycl {

platform::platform(detail::platform_state state) : reference_type(std::in_place, std::move(state)) {}

platform::platform() : platform(default_selector_v) {}

platform::platform(const detail::device_selector &selector)
    : platform(detail::select_device(selector).get_platform()) {}

platform::platform(std::shared_ptr<detail::platform_state> &&state) : reference_type(std::move(state)) {}

std::vector<device> platform::get_devices(info::device_type type) const {
    std::vector<device> result;
    std::copy_if(state().devices.begin(), state().devices.end(), std::back_inserter(result),
        [type](const device &dev) { return dev.get_info<info::device::device_type>() == type; });
    return result;
}

template<>
std::string platform::get_info<info::platform::version>() const {
    return state().config.version;
}

template<>
std::string platform::get_info<info::platform::vendor>() const {
    return state().config.vendor;
}

template<>
std::string platform::get_info<info::platform::name>() const {
    return state().config.name;
}

SIMSYCL_START_IGNORING_DEPRECATIONS
template<>
std::vector<std::string> platform::get_info<info::platform::extensions>() const {
    return state().config.extensions;
}
SIMSYCL_STOP_IGNORING_DEPRECATIONS

bool platform::has(aspect asp) const {
    return std::all_of(
        state().devices.begin(), state().devices.end(), [asp](const device &dev) { return dev.has(asp); });
}

bool platform::has_extension(const std::string &extension) const {
    return std::find(state().config.extensions.begin(), state().config.extensions.end(), extension)
        != state().config.extensions.end();
}

std::vector<platform> platform::get_platforms() { return get_system().platforms; }

void platform::add_device(const device &dev) { state().devices.push_back(dev); }

} // namespace simsycl::sycl

namespace simsycl {

sycl::platform create_platform(const platform_config &config) {
    detail::platform_state state;
    state.config = std::move(config);
    return sycl::platform(std::move(state));
}

} // namespace simsycl
