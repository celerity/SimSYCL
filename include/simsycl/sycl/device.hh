#pragma once

#include "concepts.hh" // IWYU pragma: keep
#include "enums.hh"
#include "forward.hh"
#include "info.hh"

#include "../detail/reference_type.hh"

#include <functional>
#include <string>
#include <vector>


namespace simsycl {

struct device_config;

sycl::device create_device(sycl::platform &platform, const device_config &config);

} // namespace simsycl

namespace simsycl::detail {

struct default_selector {
    int operator()(const sycl::device &device) const;
};

struct cpu_selector {
    int operator()(const sycl::device &device) const;
};

struct gpu_selector {
    int operator()(const sycl::device &device) const;
};

struct accelerator_selector {
    int operator()(const sycl::device &device) const;
};

struct device_state;

} // namespace simsycl::detail

namespace simsycl::sycl {

// Predefined device selectors
inline constexpr detail::default_selector default_selector_v;
inline constexpr detail::cpu_selector cpu_selector_v;
inline constexpr detail::gpu_selector gpu_selector_v;
inline constexpr detail::accelerator_selector accelerator_selector_v;

// Predefined types for compatibility with old SYCL 1.2.1 device selectors
using default_selector = detail::default_selector;
using cpu_selector = detail::cpu_selector;
using gpu_selector = detail::gpu_selector;
using accelerator_selector = detail::accelerator_selector;

class device final : public detail::reference_type<device, detail::device_state> {
  private:
    using reference_type = detail::reference_type<device, detail::device_state>;

  public:
    device();

    template<DeviceSelector Selector>
    explicit device(const Selector &device_selector) : device(detail::device_selector(device_selector)) {}

    bool is_cpu() const { return has(aspect::cpu); }

    bool is_gpu() const { return has(aspect::gpu); }

    bool is_accelerator() const { return has(aspect::accelerator); }

    platform get_platform() const;

    template<typename Param>
    typename Param::return_type get_info() const;

    template<typename Param>
    typename Param::return_type get_backend_info() const;

    bool has(aspect asp) const;

    [[deprecated]] bool has_extension(const std::string &extension) const;

    template<info::partition_property Prop,
        std::enable_if_t<Prop == info::partition_property::partition_equally, int> = 0>
    std::vector<device> create_sub_devices(size_t count) const;

    template<info::partition_property Prop,
        std::enable_if_t<Prop == info::partition_property::partition_by_counts, int> = 0>
    std::vector<device> create_sub_devices(const std::vector<size_t> &counts) const;

    template<info::partition_property Prop,
        std::enable_if_t<Prop == info::partition_property::partition_by_affinity_domain, int> = 0>
    std::vector<device> create_sub_devices(info::partition_affinity_domain affinity_domain) const;

    static std::vector<device> get_devices(info::device_type device_type = info::device_type::all);

  private:
    template<typename>
    friend class detail::weak_ref;

    friend device simsycl::create_device(sycl::platform &platform, const device_config &config);

    device(const detail::device_selector &selector);
    device(std::shared_ptr<detail::device_state> &&state) : reference_type(std::move(state)) {}
};


template<aspect Aspect>
struct any_device_has: std::false_type {};

template<aspect Aspect>
struct all_devices_have: std::false_type {};

template<aspect A>
inline constexpr bool any_device_has_v = any_device_has<A>::value;
template<aspect A>
inline constexpr bool all_devices_have_v = all_devices_have<A>::value;

// Returns a selector that selects a device based on desired aspects
inline auto aspect_selector(const std::vector<aspect> &aspect_list, const std::vector<aspect> &deny_list = {}) {
    return [=](const sycl::device &device) {
        for(const auto required_aspect : aspect_list) {
            if(!device.has(required_aspect)) { return -1; }
        }
        for(const auto denied_aspect : deny_list) {
            if(device.has(denied_aspect)) { return -1; }
        }
        return 0;
    };
}

template<class... AspectList>
auto aspect_selector(AspectList... aspect_list) {
    return aspect_selector(std::vector<aspect>{aspect_list...});
}

template<aspect... AspectList>
auto aspect_selector() {
    return aspect_selector(std::vector<aspect>{AspectList...});
}

} // namespace simsycl::sycl

template<>
struct std::hash<simsycl::sycl::device>
    : public std::hash<simsycl::detail::reference_type<simsycl::sycl::device, simsycl::detail::device_state>> {};
