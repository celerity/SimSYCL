#pragma once

#include "enums.hh"
#include "forward.hh"
#include "info.hh"
#include "platform.hh"

#include "../detail/reference_type.hh"

#include <string>
#include <vector>


namespace simsycl::detail {

// forward
void setup();

template<typename DeviceSelector>
sycl::device select_device(const DeviceSelector &selector);

struct default_selector {
    int operator()(const sycl::device & /* TODO */) const { return 0; }
};
struct cpu_selector : public default_selector {};         // TODO
struct gpu_selector : public default_selector {};         // TODO
struct accelerator_selector : public default_selector {}; // TODO

struct device_state {
    sycl::platform platform;
};

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

// Returns a selector that selects a device based on desired aspects
auto aspect_selector(const std::vector<aspect> &aspect_list, const std::vector<aspect> &deny_list = {});

template<class... AspectList>
auto aspect_selector(AspectList... aspect_list);

template<aspect... AspectList>
auto aspect_selector();

class device : public detail::reference_type<device, detail::device_state> {
  private:
    using reference_type = detail::reference_type<device, detail::device_state>;

  public:
    device() : device(default_selector_v) {}

    template<typename DeviceSelector>
    explicit device(const DeviceSelector &device_selector) : device(select_device(device_selector)) {}

    bool is_cpu() const;

    bool is_gpu() const;

    bool is_accelerator() const;

    platform get_platform() const { return state().platform; }

    template<typename Param>
    typename Param::return_type get_info() const {
        return {};
    }

    template<typename Param>
    typename Param::return_type get_backend_info() const {
        return {};
    }

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
    friend void detail::setup();

    device(detail::device_state state) : reference_type(std::in_place, std::move(state)) {}
};

template<aspect Aspect>
struct any_device_has;
template<aspect Aspect>
struct all_devices_have;

template<aspect A>
inline constexpr bool any_device_has_v = any_device_has<A>::value;
template<aspect A>
inline constexpr bool all_devices_have_v = all_devices_have<A>::value;

} // namespace simsycl::sycl