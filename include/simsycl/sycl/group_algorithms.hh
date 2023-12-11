#pragma once

#include "concepts.hh"
#include "group.hh"
#include "group_functions.hh"
#include "sub_group.hh"

#include "simsycl/detail/check.hh"
#include "simsycl/detail/group_operation_impl.hh"

namespace simsycl::sycl {

// any_of

template <Group G, Pointer Ptr, typename Predicate, typename T = std::remove_pointer_t<Ptr>>
    requires std::predicate<Predicate, T>
bool joint_any_of(G g, Ptr first, Ptr last, Predicate pred) {
    // approach: perform the operation sequentially on all work items, confirm that they compute the same result
    // (this is the closest we can easily get to verifying the standard requirement that
    //  "pred must be an immutable callable with the same state for all work-items in group g")

    bool result = false;
    for(auto start = first; !result && start != last; ++start) { result = pred(*start); }

    detail::perform_group_operation(g, detail::group_operation_id::joint_any_of,
        detail::group_operation_spec{//
            .init = [&]() { return std::make_unique<detail::group_joint_bool_op_data<T>>(first, last, result); },
            .reached =
                [&](detail::group_joint_bool_op_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.first == first);
                    SIMSYCL_CHECK(per_op.last == last);
                    SIMSYCL_CHECK(per_op.result == result);
                }});

    return result;
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool any_of_group(G g, T x, Predicate pred) {
    return detail::perform_group_operation(g, detail::group_operation_id::any_of,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_bool_data>(g.get_local_range().size());
                    per_op_data->values[g.get_local_linear_id()] = pred(x);
                    return per_op_data;
                },
            .reached = [&](detail::group_bool_data &per_op) { per_op.values[g.get_local_linear_id()] = pred(x); },
            .complete =
                [&](detail::group_bool_data &per_op) {
                    return std::any_of(per_op.values.begin(), per_op.values.end(), [](bool x) { return x; });
                }});
}

template <Group G>
bool any_of_group(G g, bool pred) {
    return any_of_group(g, pred, [](bool x) { return x; });
}

// all_of

template <Group G, Pointer Ptr, typename Predicate, typename T = std::remove_pointer_t<Ptr>>
    requires std::predicate<Predicate, T>
bool joint_all_of(G g, Ptr first, Ptr last, Predicate pred) {
    bool result = true;
    for(auto start = first; result && start != last; ++start) { result = pred(*start); }
    detail::perform_group_operation(g, detail::group_operation_id::joint_all_of,
        detail::group_operation_spec{//
            .init = [&]() { return std::make_unique<detail::group_joint_bool_op_data<T>>(first, last, result); },
            .reached =
                [&](detail::group_joint_bool_op_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.first == first);
                    SIMSYCL_CHECK(per_op.last == last);
                    SIMSYCL_CHECK(per_op.result == result);
                }});
    return result;
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool all_of_group(G g, T x, Predicate pred) {
    return detail::perform_group_operation(g, detail::group_operation_id::all_of,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_bool_data>(g.get_local_range().size());
                    per_op_data->values[g.get_local_linear_id()] = pred(x);
                    return per_op_data;
                },
            .reached = [&](detail::group_bool_data &per_op) { per_op.values[g.get_local_linear_id()] = pred(x); },
            .complete =
                [&](detail::group_bool_data &per_op) {
                    return std::all_of(per_op.values.begin(), per_op.values.end(), [](bool x) { return x; });
                }});
}

template <Group G>
bool all_of_group(G g, bool pred) {
    return all_of_group(g, pred, [](bool x) { return x; });
}

// none_of

template <Group G, Pointer Ptr, typename Predicate, typename T = std::remove_pointer_t<Ptr>>
    requires std::predicate<Predicate, T>
bool joint_none_of(G g, Ptr first, Ptr last, Predicate pred) {
    bool result = true;
    for(auto start = first; result && start != last; ++start) { result = !pred(*start); }
    detail::perform_group_operation(g, detail::group_operation_id::joint_none_of,
        detail::group_operation_spec{//
            .init = [&]() { return std::make_unique<detail::group_joint_bool_op_data<T>>(first, last, result); },
            .reached =
                [&](detail::group_joint_bool_op_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.first == first);
                    SIMSYCL_CHECK(per_op.last == last);
                    SIMSYCL_CHECK(per_op.result == result);
                }});
    return result;
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool none_of_group(G g, T x, Predicate pred) {
    return detail::perform_group_operation(g, detail::group_operation_id::none_of,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_bool_data>(g.get_local_range().size());
                    per_op_data->values[g.get_local_linear_id()] = pred(x);
                    return per_op_data;
                },
            .reached = [&](detail::group_bool_data &per_op) { per_op.values[g.get_local_linear_id()] = pred(x); },
            .complete =
                [&](detail::group_bool_data &per_op) {
                    return std::none_of(per_op.values.begin(), per_op.values.end(), [](bool x) { return x; });
                }});
}

template <Group G>
bool none_of_group(G g, bool pred) {
    return none_of_group(g, pred, [](bool x) { return x; });
}

// shift

template <SubGroup G, TriviallyCopyable T>
T shift_group_left(G g, T x, typename G::linear_id_type delta = 1) {
    return detail::perform_group_operation(g, detail::group_operation_id::shift_left,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_shift_data<T>>(g.get_local_range().size(), delta);
                    per_op_data->values[g.get_local_linear_id()] = x;
                    return per_op_data;
                },
            .reached =
                [&](detail::group_shift_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.delta == delta);
                    per_op.values[g.get_local_linear_id()] = x;
                },
            .complete =
                [&](detail::group_shift_data<T> &per_op) {
                    const auto target = (g.get_local_linear_id() + per_op.delta);
                    if(target >= per_op.values.size()) { return detail::unspecified<T>; }
                    return per_op.values[target];
                }});
}

template <SubGroup G, TriviallyCopyable T>
T shift_group_right(G g, T x, typename G::linear_id_type delta = 1) {
    return detail::perform_group_operation(g, detail::group_operation_id::shift_right,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_shift_data<T>>(g.get_local_range().size(), delta);
                    per_op_data->values[g.get_local_linear_id()] = x;
                    return per_op_data;
                },
            .reached =
                [&](detail::group_shift_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.delta == delta);
                    per_op.values[g.get_local_linear_id()] = x;
                },
            .complete =
                [&](detail::group_shift_data<T> &per_op) {
                    if(per_op.delta > g.get_local_linear_id()) { return detail::unspecified<T>; }
                    return per_op.values[g.get_local_linear_id() - per_op.delta];
                }});
}

// permute

template <SubGroup G, TriviallyCopyable T>
T permute_group(G g, T x, typename G::linear_id_type mask) {
    return detail::perform_group_operation(g, detail::group_operation_id::permute,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data
                        = std::make_unique<detail::group_permute_data<T>>(g.get_local_range().size(), mask);
                    per_op_data->values[g.get_local_linear_id()] = x;
                    return per_op_data;
                },
            .reached =
                [&](detail::group_permute_data<T> &per_op) {
                    SIMSYCL_CHECK(per_op.mask == mask);
                    per_op.values[g.get_local_linear_id()] = x;
                },
            .complete =
                [&](detail::group_permute_data<T> &per_op) {
                    auto target = (g.get_local_linear_id() ^ per_op.mask);
                    if(target >= per_op.values.size()) { return detail::unspecified<T>; }
                    return per_op.values[target];
                }});
}

// select

template <SubGroup G, TriviallyCopyable T>
T select_from_group(G g, T x, typename G::id_type remote_local_id) {
    return detail::perform_group_operation(g, detail::group_operation_id::select,
        detail::group_operation_spec{//
            .init =
                [&]() {
                    auto per_op_data = std::make_unique<detail::group_select_data<T>>(g.get_local_range().size());
                    per_op_data->values[g.get_local_linear_id()] = x;
                    return per_op_data;
                },
            .reached = [&](detail::group_select_data<T> &per_op) { per_op.values[g.get_local_linear_id()] = x; },
            .complete =
                [&](detail::group_select_data<T> &per_op) {
                    if(remote_local_id >= per_op.values.size()) { return detail::unspecified<T>; }
                    return per_op.values[remote_local_id];
                }});
}

// reduce


template <Group G, Pointer Ptr, SyclFunctionObject Op, typename T = std::iterator_traits<Ptr>::value_type>
T joint_reduce(G g, Ptr first, Ptr last, Op binary_op) {
    T result = *first;
    for(auto i = first + 1; first != last && i != last; ++i) { result = binary_op(result, *i); }
    simsycl::detail::joint_reduce_impl(g, first, last, {}, result);
    return result;
}

template <Group G, Pointer Ptr, Fundamental T, SyclFunctionObject Op>
T joint_reduce(G g, Ptr first, Ptr last, T init, Op binary_op) {
    T result = init;
    for(auto i = first; i != last; ++i) { result = binary_op(result, *i); }
    simsycl::detail::joint_reduce_impl(g, first, last, {init}, result);
    return result;
}

template <Group G, Fundamental T, SyclFunctionObject Op>
T reduce_over_group(G g, T x, Op binary_op) {
    return simsycl::detail::group_reduce_impl(g, x, {}, binary_op);
}

template <Group G, Fundamental V, Fundamental T, SyclFunctionObject Op>
T reduce_over_group(G g, V x, T init, Op binary_op) {
    return simsycl::detail::group_reduce_impl(g, x, {init}, binary_op);
}

// exclusive_scan

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, SyclFunctionObject Op,
    typename T = std::iterator_traits<InPtr>::value_type>
OutPtr joint_exclusive_scan(G g, InPtr first, InPtr last, OutPtr result, Op binary_op) {
    std::vector<T> results(std::distance(first, last));
    results[0] = known_identity_v<Op, T>;
    for(auto i = 0u; i < results.size() - 1; ++i) { results[i + 1] = binary_op(results[i], first[i]); }
    simsycl::detail::joint_scan_impl(
        g, simsycl::detail::group_operation_id::joint_exclusive_scan, first, last, {}, results);
    std::copy(results.cbegin(), results.cend(), result);
    return result;
}

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, Fundamental T, SyclFunctionObject Op>
OutPtr joint_exclusive_scan(G g, InPtr first, InPtr last, OutPtr result, T init, Op binary_op) {
    // CHECK first, last, result, init and the type of binary_op must be the same for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, result, init, binary_op);
}

template <Group G, Fundamental T, SyclFunctionObject Op>
T exclusive_scan_over_group(G g, T x, Op binary_op) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, binary_op);
}

template <Group G, Fundamental V, Fundamental T, SyclFunctionObject Op>
T exclusive_scan_over_group(G g, V x, T init, Op binary_op) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, init, binary_op);
}

// inclusive_scan

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, SyclFunctionObject Op,
    typename T = std::iterator_traits<InPtr>::value_type>
OutPtr joint_inclusive_scan(G g, InPtr first, InPtr last, OutPtr result, Op binary_op) {
    std::vector<T> results(std::distance(first, last));
    results[0] = *first;
    for(auto i = 1u; i < results.size(); ++i) { results[i] = binary_op(results[i - 1], first[i]); }
    simsycl::detail::joint_scan_impl(
        g, simsycl::detail::group_operation_id::joint_inclusive_scan, first, last, {}, results);
    std::copy(results.cbegin(), results.cend(), result);
    return result;
}

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, Fundamental T, SyclFunctionObject Op>
OutPtr joint_inclusive_scan(G g, InPtr first, InPtr last, OutPtr result, T init, Op binary_op) {
    // CHECK first, last, result, init and the type of binary_op must be the same for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, result, init, binary_op);
}

template <Group G, Fundamental T, SyclFunctionObject Op>
T inclusive_scan_over_group(G g, T x, Op binary_op) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, binary_op);
}

template <Group G, Fundamental V, Fundamental T, SyclFunctionObject Op>
T inclusive_scan_over_group(G g, V x, T init, Op binary_op) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, init, binary_op);
}

} // namespace simsycl::sycl
