#pragma once

#include "concepts.hh"
#include "group.hh"
#include "nd_item.hh"
#include "sub_group.hh"
#include "type_traits.hh"

#include "simsycl/detail/check.hh"
#include "simsycl/detail/group_operation_impl.hh"
#include <cstddef>
#include <memory>

namespace simsycl::sycl {

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x) {
    return group_broadcast(g, x, 0);
}

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::linear_id_type local_linear_id) {
    SIMSYCL_CHECK(local_linear_id < g.get_local_range().size());

    return perform_group_operation(g, detail::group_operation_id::broadcast,
        detail::group_operation_spec{//
            .init =
                [&](detail::group_operation_data &op_data) {
                    auto per_op_data = std::make_unique<detail::group_broadcast_data<T>>();
                    per_op_data->local_linear_id = local_linear_id;
                    per_op_data->type = std::type_index(typeid(T));
                    per_op_data->values.resize(op_data.expected_num_work_items);
                    per_op_data->values[g.get_local_linear_id()] = x;
                    return per_op_data;
                },
            .reached =
                [&](detail::group_operation_data &group_data, const detail::group_operation_data &new_data) {
                    auto &per_op = *static_cast<detail::group_broadcast_data<T> *>(group_data.per_op_data.get());
                    SIMSYCL_CHECK(per_op.local_linear_id == local_linear_id);
                    SIMSYCL_CHECK(per_op.type == std::type_index(typeid(T)));
                    SIMSYCL_CHECK(per_op.values.size() == new_data.expected_num_work_items);
                    per_op.values[g.get_local_linear_id()] = x;
                },
            .complete =
                [&](detail::group_operation_data &op_data) {
                    const auto &per_op = *static_cast<detail::group_broadcast_data<T> *>(op_data.per_op_data.get());
                    return per_op.values[local_linear_id];
                }});
}

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::id_type local_id) {
    SIMSYCL_CHECK(local_id < g.get_local_range());
    group_broadcast(g, x, local_id.get_linear_id());
}

template <Group G>
void group_barrier(G g, memory_scope fence_scope = G::fence_scope) {
    perform_group_operation(g, detail::group_operation_id::barrier,
        detail::group_operation_spec{//
            .init =
                [&](detail::group_operation_data &) {
                    auto per_op_data = std::make_unique<detail::group_barrier_data>();
                    per_op_data->fence_scope = fence_scope;
                    return per_op_data;
                },
            .reached =
                [&](detail::group_operation_data &group_data, const detail::group_operation_data &) {
                    auto &per_op = *static_cast<detail::group_barrier_data *>(group_data.per_op_data.get());
                    SIMSYCL_CHECK(per_op.fence_scope == fence_scope);
                }});
}

} // namespace simsycl::sycl
