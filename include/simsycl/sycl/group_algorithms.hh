#pragma once

#include "concepts.hh"
#include "group.hh"
#include "sub_group.hh"

#include "simsycl/detail/check.hh"

namespace simsycl::sycl {

// any_of

template <Group G, Pointer Ptr, typename Predicate>
    requires std::predicate<Predicate, std::remove_pointer_t<Ptr>>
bool joint_any_of(G g, Ptr first, Ptr last, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CEHCK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool any_of_group(G g, T x, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CHECK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G>
bool any_of_group(G g, bool pred) {
    SIMSYCL_NOT_IMPLEMENTED;
}

// all_of

template <Group G, Pointer Ptr, typename Predicate>
    requires std::predicate<Predicate, std::remove_pointer_t<Ptr>>
bool joint_all_of(G g, Ptr first, Ptr last, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CEHCK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool all_of_group(G g, T x, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CHECK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G>
bool all_of_group(G g, bool pred) {
    SIMSYCL_NOT_IMPLEMENTED;
}

// none_of

template <Group G, Pointer Ptr, typename Predicate>
    requires std::predicate<Predicate, std::remove_pointer_t<Ptr>>
bool joint_none_of(G g, Ptr first, Ptr last, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CEHCK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, typename T, typename Predicate>
    requires std::predicate<Predicate, T>
bool none_of_group(G g, T x, Predicate pred) {
    // CHECK first and last must be the same for all work-items in group g
    // CHECK pred must be an immutable callable with the same type and state for all work-items in group g
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G>
bool none_of_group(G g, bool pred) {
    SIMSYCL_NOT_IMPLEMENTED;
}

// shift

template <SubGroup G, TriviallyCopyable T>
T shift_group_left(G g, T x, typename G::linear_id_type delta = 1) {
    // CHECK delta must be the same for all work-items in the group
    SIMSYCL_NOT_IMPLEMENTED;
}

template <SubGroup G, TriviallyCopyable T>
T shift_group_right(G g, T x, typename G::linear_id_type delta = 1) {
    // CHECK delta must be the same for all work-items in the group
    SIMSYCL_NOT_IMPLEMENTED;
}

// permute

template <SubGroup G, TriviallyCopyable T>
T permute_group(G g, T x, typename G::linear_id_type mask) {
    // CHECK mask must be the same for all work-items in the group
    SIMSYCL_NOT_IMPLEMENTED;
}

// select

template <SubGroup G, TriviallyCopyable T>
T select_from_group(G g, T x, typename G::id_type remote_local_id) {
    SIMSYCL_NOT_IMPLEMENTED;
}

// reduce

template <Group G, Pointer Ptr, typename Op, typename T = std::iterator_traits<Ptr>::value_type>
    requires BinaryOperation<Op, T>
T joint_reduce(G g, Ptr first, Ptr last, Op binary_op) {
    // CHECK first and last must be the same for all work-items in group g
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, Pointer Ptr, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T joint_reduce(G g, Ptr first, Ptr last, T init, Op binary_op) {
    // CHECK first, last, init and the type of binary_op must be the same for all work-items in group g
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T reduce_over_group(G g, T x, Op binary_op) {
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, Fundamental V, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T reduce_over_group(G g, V x, T init, Op binary_op) {
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

// exclusive_scan

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, typename Op,
    typename T = std::iterator_traits<InPtr>::value_type>
    requires BinaryOperation<Op, T>
OutPtr joint_exclusive_scan(G g, InPtr first, InPtr last, OutPtr result, Op binary_op) {
    // CHECK first, last, result and the type of binary_op must be the same for all work-items in group g
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
OutPtr joint_exclusive_scan(G g, InPtr first, InPtr last, OutPtr result, T init, Op binary_op) {
    // CHECK first, last, result, init and the type of binary_op must be the same for all work-items in group g
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T exclusive_scan_over_group(G g, T x, Op binary_op) {
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, Fundamental V, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T exclusive_scan_over_group(G g, V x, T init, Op binary_op) {
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

// inclusive_scan

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, typename Op,
    typename T = std::iterator_traits<InPtr>::value_type>
    requires BinaryOperation<Op, T>
OutPtr joint_inclusive_scan(G g, InPtr first, InPtr last, OutPtr result, Op binary_op) {
    // CHECK first, last, result and the type of binary_op must be the same for all work-items in group g
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, PointerToFundamental InPtr, PointerToFundamental OutPtr, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
OutPtr joint_inclusive_scan(G g, InPtr first, InPtr last, OutPtr result, T init, Op binary_op) {
    // CHECK first, last, result, init and the type of binary_op must be the same for all work-items in group g
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T inclusive_scan_over_group(G g, T x, Op binary_op) {
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

template <Group G, Fundamental V, Fundamental T, typename Op>
    requires BinaryOperation<Op, T>
T inclusive_scan_over_group(G g, V x, T init, Op binary_op) {
    // CHECK binary_op must be an instance of a SYCL function object
    SIMSYCL_NOT_IMPLEMENTED;
}

} // namespace simsycl::sycl
