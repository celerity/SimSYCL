#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <typeindex>
#include <vector>

#include "allocation.hh"
#include "check.hh"

#include "../sycl/concepts.hh" // IWYU pragma: keep
#include "../sycl/enums.hh"
#include "../sycl/group.hh"
#include "../sycl/sub_group.hh"
#include "../sycl/vec.hh"

namespace simsycl::detail {

template<typename T>
struct unspecified_factory {
    T operator()() const
        requires(std::is_constructible_v<T, unsigned int>)
    {
        return T(0xDEADBEEF);
    }

    T operator()() const
        requires(std::is_default_constructible_v<T> && !std::is_constructible_v<T, unsigned int>)
    {
        return T();
    }
};

template<typename T, int NumElements>
    requires(std::is_constructible_v<T, unsigned int>)
struct unspecified_factory<sycl::vec<T, NumElements>> {
    sycl::vec<T, NumElements> operator()() const { return sycl::vec<T, NumElements>(T(0xDEADBEEF)); }
};

template<typename T>
T unspecified() {
    return unspecified_factory<T>{}();
}

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
    joint_reduce,
    reduce,
    joint_exclusive_scan,
    exclusive_scan,
    joint_inclusive_scan,
    inclusive_scan,
    // for verification
    exit,
};

// additional data required to implement and check correct use for some group operations

struct group_per_operation_data {
    // non-templated base class for specialized per-operation data
    virtual ~group_per_operation_data() = default;
};

template<typename T>
struct group_broadcast_data : group_per_operation_data {
    size_t local_linear_id = 0;
    std::type_index type = std::type_index(typeid(void));
    std::vector<T> values;
};
struct group_barrier_data : group_per_operation_data {
    sycl::memory_scope fence_scope;
};
template<typename T>
struct group_joint_bool_op_data : group_per_operation_data {
    T *first;
    T *last;
    bool result;
    group_joint_bool_op_data(T *first, T *last, bool result) : first(first), last(last), result(result) {}
};
struct group_bool_data : group_per_operation_data {
    std::vector<bool> values;
    group_bool_data(size_t num_work_items) : values(num_work_items) {}
};
template<typename T>
struct group_shift_data : group_per_operation_data {
    std::vector<T> values;
    size_t delta;
    group_shift_data(size_t num_work_items, size_t delta) : values(num_work_items), delta(delta) {}
};
template<typename T>
struct group_permute_data : group_per_operation_data {
    std::vector<T> values;
    size_t mask;
    group_permute_data(size_t num_work_items, size_t mask) : values(num_work_items), mask(mask) {}
};
template<typename T>
struct group_select_data : group_per_operation_data {
    std::vector<T> values;
    group_select_data(size_t num_work_items) : values(num_work_items) {}
};
template<typename T>
struct group_joint_reduce_data : group_per_operation_data {
    T *first;
    T *last;
    std::optional<T> init;
    T result;
    group_joint_reduce_data(T *first, T *last, std::optional<T> init, T result)
        : first(first), last(last), init(init), result(result) {}
};
template<typename T>
struct group_reduce_data : group_per_operation_data {
    std::optional<T> init;
    std::vector<T> values;
    group_reduce_data(size_t num_work_items, std::optional<T> init) : init(init), values(num_work_items) {}
};
template<typename T>
struct group_joint_scan_data : group_per_operation_data {
    T *first;
    T *last;
    std::optional<T> init;
    std::vector<T> results;
    group_joint_scan_data(T *first, T *last, std::optional<T> init, const std::vector<T> &results)
        : first(first), last(last), init(init), results(results) {}
};
template<typename T>
struct group_scan_data : group_per_operation_data {
    std::optional<T> init;
    std::vector<T> values;
    group_scan_data(size_t num_work_items, std::optional<T> init) : init(init), values(num_work_items) {}
};

struct group_operation_data {
    group_operation_id id;
    size_t expected_num_work_items;
    size_t num_work_items_participating;
    bool valid;
    std::unique_ptr<group_per_operation_data> per_op_data;
};

// group and sub-group impl

struct group_instance {
    size_t group_linear_id = std::numeric_limits<size_t>::max();
    std::vector<group_operation_data> operations;
    size_t num_items_exited = 0;

    group_instance() : group_linear_id(std::numeric_limits<size_t>::max()) {}
    explicit group_instance(size_t group_linear_id) : group_linear_id(group_linear_id) {}
};

struct concurrent_group {
    std::vector<concurrent_nd_item *> concurrent_nd_items;
    std::vector<allocation> local_memory_allocations;
    group_instance instance;
};

template<int Dimensions>
concurrent_group &get_concurrent_group(const sycl::group<Dimensions> &g) {
    return *g.m_concurrent_group;
}

struct sub_group_instance {
    size_t sub_group_linear_id = std::numeric_limits<size_t>::max();
    std::vector<group_operation_data> operations;

    sub_group_instance() : sub_group_linear_id(std::numeric_limits<size_t>::max()) {}
    explicit sub_group_instance(size_t sub_group_linear_id) : sub_group_linear_id(sub_group_linear_id) {}
};

struct concurrent_sub_group {
    std::vector<concurrent_nd_item *> concurrent_nd_items;
    sub_group_instance instance;
};

inline detail::concurrent_sub_group &get_concurrent_group(const sycl::sub_group &g) {
    SIMSYCL_CHECK(g.m_concurrent_group && "group operations not available in this kernel");
    return *g.m_concurrent_group;
}

void check_group_op_validity(
    int linear_id_in_group, const group_operation_data &new_op, group_operation_data &existing_op);

// group operation function template

template<typename Func>
concept GroupOpInitFunction = std::is_invocable_r_v<std::unique_ptr<group_per_operation_data>, Func>;

inline std::unique_ptr<group_per_operation_data> default_group_op_init_function() {
    return std::unique_ptr<group_per_operation_data>();
}

template<typename T>
void default_group_op_function(T &per_group) {
    (void)per_group;
}

template<GroupOpInitFunction InitF = decltype(default_group_op_init_function),
    typename PerOpT = typename std::invoke_result_t<InitF>::element_type,
    typename ReachedF = decltype(default_group_op_function<PerOpT>),
    typename CompleteF = decltype(default_group_op_function<PerOpT>)>
struct group_operation_spec {
    using per_op_t = PerOpT;
    static_assert(std::is_invocable_r_v<void, ReachedF, PerOpT &>, "reached must be of type (PerOpT&) -> void");
    static_assert(std::is_invocable_v<CompleteF, PerOpT &>, "complete must be invocable with PerOpT&");
    const InitF &init = default_group_op_init_function;
    const ReachedF &reached = default_group_op_function<PerOpT>;
    const CompleteF &complete = default_group_op_function<PerOpT>;
};

template<sycl::Group G, typename Spec>
auto perform_group_operation(G g, group_operation_id id, const Spec &spec) {
    auto &concurrent_group = detail::get_concurrent_group(g);
    auto &group_instance = concurrent_group.instance;
    const auto linear_id_in_group = g.get_local_linear_id();
    auto &this_concurrent_nd_item = *concurrent_group.concurrent_nd_items[linear_id_in_group];
    auto &this_nd_item_instance = this_concurrent_nd_item.instance;
    size_t &ops_reached
        = is_sub_group_v<G> ? this_nd_item_instance.group_ops_reached : this_nd_item_instance.sub_group_ops_reached;

    group_operation_data new_op;
    new_op.id = id;
    new_op.expected_num_work_items = g.get_local_range().size();
    new_op.num_work_items_participating = 1;
    new_op.valid = true;
    new_op.per_op_data = spec.init();

    const size_t new_op_index = ops_reached;

    if(new_op_index == group_instance.operations.size()) {
        // first item to reach this group op
        group_instance.operations.push_back(std::move(new_op));
    } else {
        // not first item to reach this group op
        SIMSYCL_CHECK(new_op_index < group_instance.operations.size() && "group operation reached in unexpected order");

        auto &op = group_instance.operations[ops_reached];
        check_group_op_validity(linear_id_in_group, new_op, op);
        spec.reached(dynamic_cast<typename Spec::per_op_t &>(*op.per_op_data));

        op.num_work_items_participating++;
    }

    ops_reached++;

    // wait for all work items to enter this group operation
    for(;;) {
        detail::yield_to_kernel_scheduler();
        // we cannot preserve a reference into `operations` across a yield since it might be resized by another item
        const auto &op = group_instance.operations[new_op_index];
        SIMSYCL_CHECK_MSG(op.valid, "group operation invalidated by another work item");
        if(op.num_work_items_participating == op.expected_num_work_items) break;
    }

    return spec.complete(dynamic_cast<typename Spec::per_op_t &>(*group_instance.operations[new_op_index].per_op_data));
}

// more specific helper functions for group operations

template<sycl::Group G, sycl::Pointer Ptr, sycl::Fundamental T>
void joint_reduce_impl(G g, Ptr first, Ptr last, std::optional<T> init, T result) {
    perform_group_operation(g, group_operation_id::joint_reduce,
        group_operation_spec{//
            .init = [&]() { return std::make_unique<group_joint_reduce_data<T>>(first, last, init, result); },
            .reached =
                [&](group_joint_reduce_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.first == first);
                    SIMSYCL_CHECK(per_op.last == last);
                    SIMSYCL_CHECK(per_op.init == init);
                    SIMSYCL_CHECK(per_op.result == result);
                }});
}

template<sycl::Group G, sycl::Fundamental T, sycl::SyclFunctionObject Op>
T group_reduce_impl(G g, T x, std::optional<T> init, Op op) {
    return perform_group_operation(g, group_operation_id::reduce,
        group_operation_spec{//
            .init =
                [&]() {
                    auto per_op = std::make_unique<group_reduce_data<T>>(g.get_local_range().size(), init);
                    per_op->values[g.get_local_linear_id()] = x;
                    return per_op;
                },
            .reached =
                [&](group_reduce_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.init == init);
                    SIMSYCL_CHECK(per_op.values.size() == g.get_local_range().size());
                    per_op.values[g.get_local_linear_id()] = x;
                },
            .complete =
                [&](const group_reduce_data<T> &per_op) {
                    T result = per_op.values.front();
                    if(init) { result = op(*init, result); }
                    if(per_op.values.size() > 1) {
                        for(auto i = per_op.values.cbegin() + 1; i != per_op.values.cend(); ++i) {
                            result = op(result, *i);
                        }
                    }
                    return result;
                }});
}

template<sycl::Group G, sycl::Pointer Ptr, sycl::Fundamental T>
void joint_scan_impl(
    G g, group_operation_id op_id, Ptr first, Ptr last, std::optional<T> init, const std::vector<T> &results) {
    perform_group_operation(g, op_id,
        group_operation_spec{//
            .init = [&]() { return std::make_unique<group_joint_scan_data<T>>(first, last, init, results); },
            .reached =
                [&](group_joint_scan_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.first == first);
                    SIMSYCL_CHECK(per_op.last == last);
                    SIMSYCL_CHECK(per_op.init == init);
                    SIMSYCL_CHECK(per_op.results == results);
                }});
}

template<sycl::Group G, sycl::Fundamental T, sycl::SyclFunctionObject Op>
T group_scan_impl(G g, group_operation_id op_id, T x, std::optional<T> init, Op op) {
    return perform_group_operation(g, op_id,
        group_operation_spec{//
            .init =
                [&]() {
                    auto per_op = std::make_unique<group_scan_data<T>>(g.get_local_range().size(), init);
                    per_op->values[g.get_local_linear_id()] = x;
                    return per_op;
                },
            .reached =
                [&](group_scan_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.init == init);
                    SIMSYCL_CHECK(per_op.values.size() == g.get_local_range().size());
                    per_op.values[g.get_local_linear_id()] = x;
                },
            .complete = [&](const group_scan_data<T> &per_op) -> T {
                std::vector<T> results(per_op.values.size());
                if(op_id == group_operation_id::exclusive_scan) {
                    results[0] = sycl::known_identity_v<Op, T>;
                    if(init) { results[0] = op(*init, results[0]); }
                    for(auto i = 0u; i < results.size() - 1; ++i) { results[i + 1] = op(results[i], per_op.values[i]); }
                } else if(op_id == group_operation_id::inclusive_scan) {
                    results[0] = per_op.values[0];
                    if(init) { results[0] = op(*init, results[0]); }
                    for(auto i = 1u; i < results.size(); ++i) { results[i] = op(results[i - 1], per_op.values[i]); }
                } else {
                    SIMSYCL_CHECK(false && "unexpected scan group operation id");
                }
                return results[g.get_local_linear_id()];
            }});
}

} // namespace simsycl::detail
