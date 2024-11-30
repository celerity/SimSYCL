#pragma once

#include "concepts.hh"
#include "enums.hh"
#include "forward.hh"
#include "info.hh"

#include "../detail/reference_type.hh"

#include <functional>
#include <string>
#include <vector>

namespace simsycl {

// forward
struct platform_config;

sycl::platform make_platform(const platform_config &config);
sycl::device make_device(sycl::platform &platform, const device_config &config);

} // namespace simsycl

namespace simsycl::detail {

class system_lock;
struct platform_state;

} // namespace simsycl::detail

namespace simsycl::sycl {

class platform final : public detail::reference_type<platform, detail::platform_state> {
  private:
    using reference_type = detail::reference_type<platform, detail::platform_state>;

  public:
    platform();

    template<DeviceSelector Selector>
    explicit platform(const Selector &device_selector) : platform(detail::device_selector(device_selector)) {}

    backend get_backend() const noexcept { return backend::simsycl; }

    std::vector<device> get_devices(info::device_type type = info::device_type::all) const;

    template<typename Param>
    typename Param::return_type get_info() const;

    template<typename Param>
    typename Param::return_type get_backend_info() const;

    bool has(aspect asp) const;

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL bool has_extension(const std::string &extension) const;

    static std::vector<platform> get_platforms();

  private:
    template<typename>
    friend class detail::weak_ref;

    friend sycl::platform simsycl::make_platform(const platform_config &config);
    friend device simsycl::make_device(platform &platform, const device_config &config);

    platform(const detail::device_selector &selector);
    platform(std::shared_ptr<detail::platform_state> &&state) : reference_type(std::move(state)) {}

    void add_device(const device &dev, detail::system_lock &lock);
};

} // namespace simsycl::sycl

template<>
struct std::hash<simsycl::sycl::platform>
    : std::hash<simsycl::detail::reference_type<simsycl::sycl::platform, simsycl::detail::platform_state>> {};
