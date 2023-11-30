#pragma once

#include "concepts.hh"
#include "group.hh"
#include "group_functions.hh"
#include "sub_group.hh"

#include "simsycl/detail/check.hh"

namespace simsycl::sycl {

// any_of

template <Group G, Pointer Ptr, typename Predicate>
    requires std::predicate<Predicate, std::remove_pointer_t<Ptr>>
bool joint_any_of(G g, Ptr first, Ptr last, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CEHCK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, pred);
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool any_of_group(G g, T x, Predicate pred) {
    // CHECK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, pred);
}

template <Group G>
bool any_of_group(G g, bool pred) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, pred);
}

// all_of

template <Group G, Pointer Ptr, typename Predicate>
    requires std::predicate<Predicate, std::remove_pointer_t<Ptr>>
bool joint_all_of(G g, Ptr first, Ptr last, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CEHCK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, pred);
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool all_of_group(G g, T x, Predicate pred) {
    // CHECK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, pred);
}

template <Group G>
bool all_of_group(G g, bool pred) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, pred);
}

// none_of

template <Group G, Pointer Ptr, typename Predicate>
    requires std::predicate<Predicate, std::remove_pointer_t<Ptr>>
bool joint_none_of(G g, Ptr first, Ptr last, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CEHCK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, pred);
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool none_of_group(G g, T x, Predicate pred) {
    // CHECK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, pred);
}

template <Group G>
bool none_of_group(G g, bool pred) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, pred);
}

// shift

template <SubGroup G, TriviallyCopyable T>
T shift_group_left(G g, T x, typename G::linear_id_type delta = 1) {
    // CHECK delta must be the same for all work-items in the group
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, delta);
}

template <SubGroup G, TriviallyCopyable T>
T shift_group_right(G g, T x, typename G::linear_id_type delta = 1) {
    // CHECK delta must be the same for all work-items in the group
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, delta);
}

// permute

template <SubGroup G, TriviallyCopyable T>
T permute_group(G g, T x, typename G::linear_id_type mask) {
    // CHECK mask must be the same for all work-items in the group
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, mask);
}

// select

template <SubGroup G, TriviallyCopyable T>
T select_from_group(G g, T x, typename G::id_type remote_local_id) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, remote_local_id);
}

// reduce

template <Group G, Pointer Ptr, typename Op, typename T = std::iterator_traits<Ptr>::value_type>
    requires BinaryOperation<Op, T>
T joint_reduce(G g, Ptr first, Ptr last, Op binary_op) {
    // CHECK first and last must be the same for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, binary_op);
}

template <Group G, Pointer Ptr, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T joint_reduce(G g, Ptr first, Ptr last, T init, Op binary_op) {
    // CHECK first, last, init and the type of binary_op must be the same for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, init, binary_op);
}

template <Group G, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T reduce_over_group(G g, T x, Op binary_op) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, binary_op);
}

template <Group G, Fundamental V, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T reduce_over_group(G g, V x, T init, Op binary_op) {
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, x, init, binary_op);
}

// exclusive_scan

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, SyclFunctionObject Op,
    typename T = std::iterator_traits<InPtr>::value_type>
OutPtr joint_exclusive_scan(G g, InPtr first, InPtr last, OutPtr result, Op binary_op) {
    // CHECK first, last, result and the type of binary_op must be the same for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, result, binary_op);
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
    // CHECK first, last, result and the type of binary_op must be the same for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(g, first, last, result, binary_op);
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
