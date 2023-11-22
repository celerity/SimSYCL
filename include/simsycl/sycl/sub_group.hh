#pragma once

#include "forward.hh"
#include "id.hh"
#include "range.hh"
#include "simsycl/detail/check.hh"

namespace simsycl::sycl {

class sub_group {
  public:
    using id_type = id<1>;
    using range_type = range<1>;
    using linear_id_type = uint32_t;
    static constexpr int dimensions = 1;
    static constexpr sycl::memory_scope fence_scope = sycl::memory_scope::sub_group;

    id_type get_local_id() const { SIMSYCL_NOT_IMPLEMENTED; }

    linear_id_type get_local_linear_id() const { SIMSYCL_NOT_IMPLEMENTED; }

    range_type get_local_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    range_type get_max_local_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    id_type get_group_id() const { SIMSYCL_NOT_IMPLEMENTED; }

    linear_id_type get_group_linear_id() const { SIMSYCL_NOT_IMPLEMENTED; }

    range_type get_group_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    template <typename T>
    T shuffle(T x, id_type local_id) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T>
    T shuffle_down(T x, uint32_t delta) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T>
    T shuffle_up(T x, uint32_t delta) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T>
    T shuffle_xor(T x, id_type value) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename CVT, typename T = std::remove_cv_t<CVT>>
    T load(CVT *src) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T>
    void store(T *dst, const T &x) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    // synchronization functions

    void barrier() const { SIMSYCL_NOT_IMPLEMENTED; }

    [[deprecated("use barrier() without a fence_space")]] void barrier(access::fence_space) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    // deprecated collective functions
    template <typename T>
    [[deprecated("use freestanding function instead")]] T broadcast(T x, id<1> local_id) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function instead")]] T reduce(T x, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function instead")]] T reduce(T x, T init, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function instead")]] T exclusive_scan(T x, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function instead")]] T exclusive_scan(T x, T init, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function instead")]] T inclusive_scan(T x, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename T, class BinaryOperation>
    [[deprecated("use freestanding function instead")]] T inclusive_scan(T x, T init, BinaryOperation op) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    linear_id_type get_group_linear_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    linear_id_type get_local_linear_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    bool leader() const { return get_local_linear_id() == 0; }

    friend bool operator==(const sub_group &lhs, const sub_group &rhs) {
        return lhs.get_group_id() == rhs.get_group_id();
    }

    friend bool operator!=(const sub_group &lhs, const sub_group &rhs) { return !(lhs == rhs); }
};

} // namespace simsycl::sycl
