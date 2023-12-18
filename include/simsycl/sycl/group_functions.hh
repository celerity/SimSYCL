#pragma once

#include "simsycl/detail/check.hh"
#include "simsycl/detail/group_operation_impl.hh"
#include "simsycl/detail/nd_memory.hh"

#include <memory>

namespace simsycl::sycl {

template<Group G, TriviallyCopyable T>
T group_broadcast(G g, T x) {
    return group_broadcast(g, x, 0);
}

template<Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::linear_id_type local_linear_id) {
    SIMSYCL_CHECK(local_linear_id < g.get_local_range().size());

    return perform_group_operation(g, detail::group_operation_id::broadcast,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_broadcast_data<T>>();
                    per_op_data->local_linear_id = local_linear_id;
                    per_op_data->type = std::type_index(typeid(T));
                    per_op_data->values.resize(g.get_local_range().size());
                    per_op_data->values[g.get_local_linear_id()] = x;
                    return per_op_data;
                },
            .reached =
                [&](detail::group_broadcast_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.local_linear_id == local_linear_id);
                    SIMSYCL_CHECK(per_op.type == std::type_index(typeid(T)));
                    SIMSYCL_CHECK(per_op.values.size() == g.get_local_range().size());
                    per_op.values[g.get_local_linear_id()] = x;
                },
            .complete = [&](const detail::group_broadcast_data<T> &per_op) { return per_op.values[local_linear_id]; }});
}

template<Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::id_type local_id) {
    SIMSYCL_CHECK(all_true(local_id < g.get_local_range()));
    group_broadcast(g, x, detail::get_linear_index(g.get_local_range(), local_id));
}

template<Group G>
void group_barrier(G g, memory_scope fence_scope = G::fence_scope) {
    perform_group_operation(g, detail::group_operation_id::barrier,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_barrier_data>();
                    per_op_data->fence_scope = fence_scope;
                    return per_op_data;
                },
            .reached = [&](detail::group_barrier_data &per_op) { SIMSYCL_CHECK(per_op.fence_scope == fence_scope); }});
}

} // namespace simsycl::sycl
