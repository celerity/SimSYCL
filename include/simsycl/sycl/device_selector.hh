#pragma once

#include "device.hh"

namespace simsycl::sycl {

class device_selector {
  public:
    device_selector() {}

    device_selector(const device_selector &rhs) { (void)rhs; };

    device_selector &operator=(const device_selector &rhs) {
        (void)rhs;
        return *this;
    };

    virtual ~device_selector() {}

    device select_device() const { return {}; }

    virtual int operator()(const device &device) const = 0;
};

} // namespace simsycl::sycl
