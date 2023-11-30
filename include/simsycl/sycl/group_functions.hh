#pragma once

#include "concepts.hh"
#include "group.hh"
#include "nd_item.hh"
#include "sub_group.hh"

#include "simsycl/detail/check.hh"

namespace simsycl::sycl {

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x) {
    return group_broadcast(g, x, 0);
}

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::linear_id_type local_linear_id) {
    SIMSYCL_CHECK(local_linear_id < g.get_local_range().size());

    auto &group_impl = detail::get_group_impl(g);
    const auto linear_id_in_group = g.get_local_linear_id();
    auto &this_nd_item = *group_impl.item_impls[linear_id_in_group];
    size_t &ops_reached
        = std::is_same_v<G, sub_group> ? this_nd_item.group_ops_reached : this_nd_item.sub_group_ops_reached;

    detail::group_operation_data new_op;
    new_op.id = detail::group_operation_id::broadcast;
    new_op.expected_num_work_items = g.get_local_range().size();
    new_op.num_work_items_participating = 1;
    auto per_op_data = std::make_unique<detail::group_broadcast_data<T>>();
    per_op_data->local_linear_id = local_linear_id;
    per_op_data->type = std::type_index(typeid(T));
    per_op_data->values.resize(new_op.expected_num_work_items);
    per_op_data->values[linear_id_in_group] = x;
    new_op.per_op_data = std::move(per_op_data);

    if(ops_reached == group_impl.operations.size()) {
        // first item to reach this group op
        ops_reached++;
        group_impl.operations.push_back(std::move(new_op));
    } else if(ops_reached < group_impl.operations.size()) {
        // not first item to reach this group op
        auto &op = group_impl.operations[ops_reached];
        SIMSYCL_CHECK(op.id == new_op.id);
        SIMSYCL_CHECK(op.expected_num_work_items == new_op.expected_num_work_items);
        SIMSYCL_CHECK(op.num_work_items_participating < op.expected_num_work_items);
        auto &per_op = *static_cast<detail::group_broadcast_data<T> *>(op.per_op_data.get());
        SIMSYCL_CHECK(per_op.local_linear_id == local_linear_id);
        SIMSYCL_CHECK(per_op.type == std::type_index(typeid(T)));
        SIMSYCL_CHECK(per_op.values.size() == op.expected_num_work_items);
        per_op.values[linear_id_in_group] = x;
        ops_reached++;

        op.num_work_items_participating++;
        if(op.num_work_items_participating == op.expected_num_work_items) {
            // last item to reach this group op, but no operation required for broadcast
        }
    } else {
        SIMSYCL_CHECK(false && "group operation reached in unexpected order");
    }
    this_nd_item.barrier();

    // broadcast is complete, return the value
    const auto &op = group_impl.operations[ops_reached - 1];
    const auto &per_op = *static_cast<detail::group_broadcast_data<T> *>(op.per_op_data.get());
    return per_op.values[local_linear_id];
}

template <Group G, TriviallyCopyable T>
T group_broadcast(G g, T x, typename G::id_type local_id) {
    SIMSYCL_CHECK(local_id < g.get_local_range());
    group_broadcast(g, x, local_id.get_linear_id());
}

template <Group G>
void group_barrier(G g, memory_scope fence_scope = G::fence_scope) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, fence_scope);
}

} // namespace simsycl::sycl
