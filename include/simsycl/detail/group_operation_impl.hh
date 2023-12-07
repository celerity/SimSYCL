#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "simsycl/sycl/concepts.hh"
#include "simsycl/sycl/enums.hh"
#include "simsycl/sycl/group.hh"
#include "simsycl/sycl/sub_group.hh"

namespace simsycl::detail {

template <typename T>
constexpr auto unspecified = (T)(0xDEADBEEFull);

enum class group_operation_id {
    broadcast,
    barrier,
    joint_any_of,
    any_of,
    joint_all_of,
    all_of,
    joint_none_of,
    none_of,
    shift_left,
    shift_right,
    permute,
    select,
    reduce,
    joint_exclusive_scan,
    exclusive_scan,
    joint_inclusive_scan,
    inclusive_scan,
};

// additional data required to implement and check correct use for some group operations

struct group_per_operation_data {
    // non-templated base class for specialized per-operation data
    virtual ~group_per_operation_data() = default;
};

template <typename T>
struct group_broadcast_data : group_per_operation_data {
    size_t local_linear_id = 0;
    std::type_index type = std::type_index(typeid(void));
    std::vector<T> values;
};
struct group_barrier_data : group_per_operation_data {
    sycl::memory_scope fence_scope;
};
struct group_joint_op_data : group_per_operation_data {
    std::intptr_t first;
    std::intptr_t last;
    bool result;
};
struct group_bool_data : group_per_operation_data {
    std::vector<bool> values;
};
template <typename T>
struct group_shift_data : group_per_operation_data {
    std::vector<T> values;
    size_t delta;
    group_shift_data(size_t num_work_items, size_t delta) : values(num_work_items), delta(delta) {}
};
template <typename T>
struct group_permute_data : group_per_operation_data {
    std::vector<T> values;
    size_t mask;
    group_permute_data(size_t num_work_items, size_t mask) : values(num_work_items), mask(mask) {}
};
struct group_joint_scan_data : group_per_operation_data {
    std::intptr_t first;
    std::intptr_t last;
    std::intptr_t result;
    std::vector<std::byte> init;
};

struct group_operation_data {
    group_operation_id id;
    size_t expected_num_work_items;
    size_t num_work_items_participating;
    std::unique_ptr<group_per_operation_data> per_op_data;
};

// group and sub-group impl

struct group_impl {
    std::vector<nd_item_impl *> item_impls;
    std::vector<group_operation_data> operations;
};
template <int Dimensions>
group_impl &get_group_impl(const sycl::group<Dimensions> &g) {
    return *g.m_impl;
}

struct sub_group_impl {
    std::vector<nd_item_impl *> item_impls;
    std::vector<group_operation_data> operations;
};
detail::sub_group_impl &get_group_impl(const sycl::sub_group &g) { return *g.m_impl; }

// group operation function template

template <typename Func>
concept GroupOpInitFunction = std::is_invocable_r_v<std::unique_ptr<group_per_operation_data>, Func>;
std::unique_ptr<group_per_operation_data> default_group_op_init_function() {
    return std::unique_ptr<group_per_operation_data>();
}

template <typename T>
void default_group_op_function(T &per_group) {
    (void)per_group;
}

template <GroupOpInitFunction InitF = decltype(default_group_op_init_function),
    typename PerOpT = std::invoke_result_t<InitF>::element_type,
    typename ReachedF = decltype(default_group_op_function<PerOpT>),
    typename CompleteF = decltype(default_group_op_function<PerOpT>)>
struct group_operation_spec {
    using per_op_t = PerOpT;
    const InitF &init = default_group_op_init_function;
    const ReachedF &reached = default_group_op_function<PerOpT>;
    const CompleteF &complete = default_group_op_function<PerOpT>;
};

template <sycl::Group G, typename Spec>
auto perform_group_operation(G g, group_operation_id id, const Spec &spec) {
    auto &group_impl = detail::get_group_impl(g);
    const auto linear_id_in_group = g.get_local_linear_id();
    auto &this_nd_item_impl = *group_impl.item_impls[linear_id_in_group];
    size_t &ops_reached
        = sycl::is_sub_group_v<G> ? this_nd_item_impl.group_ops_reached : this_nd_item_impl.sub_group_ops_reached;

    detail::group_operation_data new_op;
    new_op.id = id;
    new_op.expected_num_work_items = g.get_local_range().size();
    new_op.num_work_items_participating = 1;
    new_op.per_op_data = spec.init();

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
        spec.reached(dynamic_cast<Spec::per_op_t &>(*op.per_op_data));
        ops_reached++;

        op.num_work_items_participating++;
        if(op.num_work_items_participating == op.expected_num_work_items) {
            // last item to reach this group op
        }
    } else {
        SIMSYCL_CHECK(false && "group operation reached in unexpected order");
    }
    this_nd_item_impl.barrier();

    return spec.complete(dynamic_cast<Spec::per_op_t &>(*group_impl.operations[ops_reached - 1].per_op_data));
}

} // namespace simsycl::detail
