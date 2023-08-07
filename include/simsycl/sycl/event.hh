#pragma once

#include "forward.hh"

namespace simsycl::detail {
sycl::event make_event();
}

namespace simsycl::sycl {

class event {
  private:
    event() = default;
    friend event detail::make_event();
};

} // namespace simsycl::sycl

namespace simsycl::detail {
sycl::event make_event() { return {}; }
} // namespace simsycl::detail
