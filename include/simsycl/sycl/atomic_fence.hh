#pragma once

#include "enums.hh"

namespace simsycl::sycl {

inline void atomic_fence(memory_order order, memory_scope scope) {
    (void)order;
    (void)scope;
    // TODO yield if order != relaxed and this is inside an nd_range kernel
}

} // namespace simsycl::sycl
