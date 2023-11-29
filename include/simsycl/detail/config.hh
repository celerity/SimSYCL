#pragma once

#include <inttypes.h>

namespace simsycl::detail {

class config {
  public:
    static constexpr uint32_t max_sub_group_size = 32;
};

} // namespace simsycl::detail
