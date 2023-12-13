#pragma once

#include "device.hh"
#include "enums.hh"
#include "forward.hh"
#include "info.hh"

#include "../detail/reference_type.hh"

#include <string>
#include <vector>


namespace simsycl::detail {

struct platform_state {};

} // namespace simsycl::detail

namespace simsycl::sycl {

class platform : public detail::reference_type<platform, detail::platform_state> {
  private:
    using reference_type = detail::reference_type<platform, detail::platform_state>;

  public:
    platform() : platform(default_selector_v) {}

    template <typename DeviceSelector>
    explicit platform(const DeviceSelector &device_selector) : reference_type(std::in_place) {
        (void)(device_selector);
    }

    backend get_backend() const noexcept;

    std::vector<device> get_devices(info::device_type = info::device_type::all) const;

    template <typename Param>
    typename Param::return_type get_info() const {
        return {};
    }

    template <typename Param>
    typename Param::return_type get_backend_info() const {
        return {};
    }

    bool has(aspect asp) const;

    [[deprecated]] bool has_extension(const std::string &extension) const;

    static std::vector<platform> get_platforms();
};

} // namespace simsycl::sycl
