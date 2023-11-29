#pragma once

#include "forward.hh"

#include "id.hh"
#include "range.hh"
#include "type_traits.hh"

#include "simsycl/detail/check.hh"

namespace simsycl::detail {

struct group_impl {};

template <int Dimensions>
sycl::group<Dimensions> make_group(const sycl::id<Dimensions> &index, const sycl::range<Dimensions> &global_range,
    const sycl::range<Dimensions> &local_range, const sycl::range<Dimensions> &group_range, group_impl *impl) {
    return sycl::group<Dimensions>(index, global_range, local_range, group_range, impl);
}

} // namespace simsycl::detail


namespace simsycl::sycl {

template <int Dimensions>
class group {
  public:
    using id_type = id<Dimensions>;
    using range_type = range<Dimensions>;
    using linear_id_type = size_t;
    static constexpr int dimensions = Dimensions;
    static constexpr sycl::memory_scope fence_scope = sycl::memory_scope::work_group;

    group() = delete;

    [[deprecated("use sycl::group::get_group_id")]] id<Dimensions> get_id() const { return m_index; }

    [[deprecated("use sycl::group::get_group_id")]] size_t get_id(int dimension) const { return m_index[dimension]; }

    id<Dimensions> get_group_id() const { return m_index; }

    size_t get_group_id(int dimension) const { return m_index[dimension]; }

    [[deprecated("deprecated in SYCL2020")]] range<Dimensions> get_global_range() const { return m_global_range; }

    size_t get_global_range(int dimension) const { return m_global_range[dimension]; }

    id<Dimensions> get_local_id() const { SIMSYCL_NOT_IMPLEMENTED; }

    size_t get_local_id(int dimension) const { return get_local_id()[dimension]; }

    size_t get_local_linear_id() const { SIMSYCL_NOT_IMPLEMENTED; }

    range<Dimensions> get_local_range() const { return m_local_range; }

    size_t get_local_range(int dimension) const { return m_local_range[dimension]; }

    size_t get_local_linear_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    range<Dimensions> get_group_range() const { return m_group_range; }

    size_t get_group_range(int dimension) const { return get_group_range()[dimension]; }

    size_t get_group_linear_range() const { SIMSYCL_NOT_IMPLEMENTED; }

    range<Dimensions> get_max_local_range() const { return get_local_range(); }

    size_t operator[](int dimension) const { return m_index[dimension]; }

    [[deprecated("use sycl::group::get_group_linear_id")]] size_t get_linear_id() const {
        return get_group_linear_id();
    }

    size_t get_group_linear_id() const { SIMSYCL_NOT_IMPLEMENTED; }

    bool leader() const { return (get_local_linear_id() == 0); }

    template <typename WorkItemFunctionT>
    void parallel_for_work_item(WorkItemFunctionT func) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename WorkItemFunctionT>
    void parallel_for_work_item(range<Dimensions> flexible_range, WorkItemFunctionT func) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <access::mode AccessMode = access_mode::read_write>
    void mem_fence(typename std::enable_if_t<AccessMode == access_mode::read || AccessMode == access_mode::write
                           || AccessMode == access_mode::read_write,
                       access::fence_space>
                       access_space
        = access::fence_space::global_and_local) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename... Events>
    void wait_for(Events... events) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    bool operator==(const group<Dimensions> &rhs) const {
        return (rhs.m_global_range == m_global_range) && (rhs.m_local_range == m_local_range)
            && (rhs.m_index == m_index);
    }

    bool operator!=(const group<Dimensions> &rhs) const { return !((*this) == rhs); }

  private:
    id<Dimensions> m_index;
    range<Dimensions> m_global_range;
    range<Dimensions> m_local_range;
    range<Dimensions> m_group_range;
    detail::group_impl *m_impl;

    group(const id<Dimensions> &index, const range<Dimensions> &global_range, const range<Dimensions> &local_range,
        const range<Dimensions> &group_range, detail::group_impl *impl)
        : m_index(index), m_global_range(global_range), m_local_range(local_range), m_group_range(group_range),
          m_impl(impl) {}

    friend group<Dimensions> detail::make_group<Dimensions>(const sycl::id<Dimensions> &index,
        const sycl::range<Dimensions> &global_range, const sycl::range<Dimensions> &local_range,
        const sycl::range<Dimensions> &group_range, detail::group_impl *impl);
};

template <int Dimensions>
struct is_group<group<Dimensions>> : std::true_type {};

} // namespace simsycl::sycl
