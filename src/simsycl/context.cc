#include "simsycl/sycl/context.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/info.hh"
#include "simsycl/sycl/platform.hh"


namespace simsycl::detail {

template<typename Info>
typename Info::return_type common_capabilities(const std::vector<sycl::device> &devices) {
    SIMSYCL_CHECK(!devices.empty()); // TODO throw instead
    auto common_caps = devices[0].get_info<Info>();
    for(size_t i = 1; i < devices.size(); ++i) {
        const auto caps = devices[i].get_info<Info>();
        const auto last = std::remove_if(common_caps.begin(), common_caps.end(),
            [&](const auto c) { return std::find(caps.begin(), caps.end(), c) == caps.end(); });
        common_caps.erase(last, common_caps.end());
    }
    return common_caps;
}

struct context_state {
    sycl::platform platform;
    std::vector<sycl::device> devices;
    sycl::async_handler async_handler;

    std::vector<sycl::memory_order> atomic_memory_order_capabilities;
    std::vector<sycl::memory_order> atomic_fence_order_capabilities;
    std::vector<sycl::memory_scope> atomic_memory_scope_capabilities;
    std::vector<sycl::memory_scope> atomic_fence_scope_capabilities;

    context_state(const sycl::platform &platform, const std::vector<sycl::device> &devices,
        const sycl::async_handler &async_handler)
        : platform(platform), devices(devices), async_handler(async_handler),
          atomic_memory_order_capabilities(
              common_capabilities<sycl::info::device::atomic_memory_order_capabilities>(devices)),
          atomic_fence_order_capabilities(
              common_capabilities<sycl::info::device::atomic_fence_order_capabilities>(devices)),
          atomic_memory_scope_capabilities(
              common_capabilities<sycl::info::device::atomic_memory_scope_capabilities>(devices)),
          atomic_fence_scope_capabilities(
              common_capabilities<sycl::info::device::atomic_fence_scope_capabilities>(devices)) {}
};

sycl::platform get_common_platform(const std::vector<sycl::device> &devices) {
    SIMSYCL_CHECK(!devices.empty()); // TODO throw instead
    const auto common = devices[0].get_platform();
    for(size_t i = 1; i < devices.size(); ++i) { SIMSYCL_CHECK(devices[i].get_platform() == common); }
    return common;
}

} // namespace simsycl::detail


namespace simsycl::sycl {

context::context(internal_t /* tag */, const std::vector<device> &devices, const async_handler &async_handler,
    const property_list &prop_list)
    : reference_type(std::in_place, get_common_platform(devices), devices, async_handler),
      property_interface(prop_list, property_compatibility{}) {}

context::context(const property_list &prop_list) : context(internal, {}, {}, prop_list) {}

context::context(async_handler async_handler, const property_list &prop_list)
    : context(internal, {}, async_handler, prop_list) {}

context::context(const device &dev, const property_list &prop_list) : context(internal, {dev}, {}, prop_list) {}

context::context(const device &dev, async_handler async_handler, const property_list &prop_list)
    : context(internal, {dev}, async_handler, prop_list) {}

context::context(const std::vector<device> &device_list, const property_list &prop_list)
    : context(internal, device_list, {}, prop_list) {}

context::context(const std::vector<device> &device_list, async_handler async_handler, const property_list &prop_list)
    : context(internal, device_list, async_handler, prop_list) {}

template<>
platform context::get_info<info::context::platform>() const {
    return state().platform;
}

template<>
std::vector<device> context::get_info<info::context::devices>() const {
    return state().devices;
}

template<>
std::vector<sycl::memory_order> context::get_info<info::context::atomic_memory_order_capabilities>() const {
    return state().atomic_memory_order_capabilities;
}

template<>
std::vector<sycl::memory_order> context::get_info<info::context::atomic_fence_order_capabilities>() const {
    return state().atomic_fence_order_capabilities;
}

template<>
std::vector<sycl::memory_scope> context::get_info<info::context::atomic_memory_scope_capabilities>() const {
    return state().atomic_memory_scope_capabilities;
}

template<>
std::vector<sycl::memory_scope> context::get_info<info::context::atomic_fence_scope_capabilities>() const {
    return state().atomic_fence_scope_capabilities;
}

platform context::get_platform() const { return get_info<info::context::platform>(); }

std::vector<device> context::get_devices() const { return state().devices; }

} // namespace simsycl::sycl
