#pragma once

#include "enums.hh"
#include "forward.hh"

#include <string>
#include <vector>

namespace simsycl::sycl {

class platform {
  public:
    platform();

    template <typename DeviceSelector>
    explicit platform(const DeviceSelector &device_selector);

    /* -- common interface members -- */

    backend get_backend() const noexcept;

    std::vector<device> get_devices(info::device_type = info::device_type::all) const;

    template <typename Param>
    typename Param::return_type get_info() const;

    template <typename Param>
    typename Param::return_type get_backend_info() const;

    bool has(aspect asp) const;

    [[deprecated]] bool has_extension(const std::string &extension) const;

    static std::vector<platform> get_platforms();
};

} // namespace simsycl::sycl
