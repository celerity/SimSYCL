#pragma once

#include "forward.hh"
#include "id.hh"
#include "range.hh"
#include "type_traits.hh"

#include "simsycl/detail/check.hh"
#include "simsycl/detail/config.hh"
#include "simsycl/detail/group_operation_impl.hh"

namespace simsycl::sycl {

class sub_group {
  public:
    using id_type = id<1>;
    using range_type = range<1>;
    using linear_id_type = uint32_t;
    static constexpr int dimensions = 1;
    static constexpr sycl::memory_scope fence_scope = sycl::memory_scope::sub_group;

    id_type get_local_id() const { return m_local_id; }

    linear_id_type get_local_linear_id() const { return (uint32_t)(size_t)m_local_id; }

    range_type get_local_range() const { return m_local_range; }

    range_type get_max_local_range() const { return range<1>(detail::config::max_sub_group_size); }

    id_type get_group_id() const { return m_group_id; }

    linear_id_type get_group_linear_id() const { return (uint32_t)(size_t)m_group_id; }

    range_type get_group_range() const { return m_group_range; }

    // synchronization functions

    [[deprecated("use freestanding function group_barrier instead")]] void barrier() const { SIMSYCL_NOT_IMPLEMENTED; }

    [[deprecated("use freestanding function group_barrier instead")]] void barrier(access::fence_space space) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(space);
    }

    // deprecated collective functions
    template <typename T>
    [[deprecated("use freestanding function group_broadcast instead")]] T broadcast(T x, id<1> local_id) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(x, local_id);
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function group_reduce instead")]] T reduce(T x, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(x, op);
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function group_reduce instead")]] T reduce(T x, T init, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(x, init, op);
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function exclusive_scan_over_group instead")]] T exclusive_scan(
        T x, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(x, op);
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function exclusive_scan_over_group instead")]] T exclusive_scan(
        T x, T init, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(x, init, op);
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function inclusive_scan_over_group instead")]] T inclusive_scan(
        T x, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(x, op);
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function inclusive_scan_over_group instead")]] T inclusive_scan(
        T x, T init, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(x, init, op);
    }

    linear_id_type get_group_linear_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    linear_id_type get_local_linear_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    bool leader() const { return get_local_linear_id() == 0; }

    friend bool operator==(const sub_group &lhs, const sub_group &rhs) {
        return lhs.get_group_id() == rhs.get_group_id();
    }

    friend bool operator!=(const sub_group &lhs, const sub_group &rhs) { return !(lhs == rhs); }

  private:
    id_type m_local_id;
    range_type m_local_range;
    id_type m_group_id;
    range_type m_group_range;

    detail::sub_group_impl *m_impl;

    sub_group(const id_type &local_id, const range_type &local_range, const id_type &group_id,
        const range_type &group_range, detail::sub_group_impl *impl)
        : m_local_id(local_id), m_local_range(local_range), m_group_id(group_id), m_group_range(group_range),
          m_impl(impl) {}

    friend sycl::sub_group detail::make_sub_group(const sycl::id<1> &local_id, const sycl::range<1> &local_range,
        const sycl::id<1> &group_id, const sycl::range<1> &group_range, detail::sub_group_impl *impl);

    friend detail::sub_group_impl &detail::get_group_impl(sycl::sub_group &g);
};

template <>
struct is_group<sub_group> : std::true_type {};

template <>
struct is_sub_group<sub_group> : std::true_type {};

} // namespace simsycl::sycl

namespace simsycl::detail {

struct sub_group_impl {
    std::vector<nd_item_impl *> item_impls;
    std::vector<group_operation_data> operations;
};

sycl::sub_group make_sub_group(const sycl::id<1> &local_id, const sycl::range<1> &local_range,
    const sycl::id<1> &group_id, const sycl::range<1> &group_range, detail::sub_group_impl *impl) {
    return sycl::sub_group(local_id, local_range, group_id, group_range, impl);
}

detail::sub_group_impl &get_group_impl(sycl::sub_group &g) { return *g.m_impl; }

} // namespace simsycl::detail
