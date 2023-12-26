#pragma once

#include "enums.hh"
#include "forward.hh"


namespace simsycl::sycl {

inline void atomic_fence(memory_order order, memory_scope scope) {
    (void)order;
    (void)scope;
    // Guarantee forward progress in kernels that use atomics for synchronization. It is somewhat unclear to me whether
    // that is strictly necessary if order == relaxed since that does not introduce any ordering.
    detail::maybe_yield_to_kernel_scheduler();
}

} // namespace simsycl::sycl
