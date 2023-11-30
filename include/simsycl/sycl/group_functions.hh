#pragma once

#include "concepts.hh"
#include "group.hh"
#include "sub_group.hh"

#include "simsycl/detail/check.hh"

namespace simsycl::sycl {

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x);
}

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::linear_id_type local_linear_id) {
    // CHECK local_linear_id must be the same for all work-items in the group
    SIMSYCL_CHECK(local_linear_id < g.get_local_range().size());
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, local_linear_id);
}

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::id_type local_id) {
    // CHECK local_id must be the same for all work-items in the group
    SIMSYCL_CHECK(local_id < g.get_local_range());
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, local_id);
}

template <Group G>
void group_barrier(G g, memory_scope fence_scope = G::fence_scope) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, fence_scope);
}

} // namespace simsycl::sycl
