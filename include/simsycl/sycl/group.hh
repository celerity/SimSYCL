#pragma once

#include "forward.hh"

#include "id.hh"
#include "range.hh"
#include "type_traits.hh"

#include "simsycl/detail/check.hh"
#include "simsycl/detail/group_operation_impl.hh"

namespace simsycl::detail {

template <int Dimensions>
sycl::group<Dimensions> make_group(const sycl::item<Dimensions, false> &local_item,
    const sycl::item<Dimensions, false> &global_item, const sycl::item<Dimensions, false> &group_item,
    detail::group_impl *impl) {
    return sycl::group<Dimensions>(local_item, global_item, group_item, impl);
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

    [[deprecated("use sycl::group::get_group_id")]] id_type get_id() const { return get_group_id(); }

    [[deprecated("use sycl::group::get_group_id")]] size_t get_id(int dimension) const {
        return get_group_id(dimension);
    }

    id_type get_group_id() const { return m_group_item.get_id(); }

    size_t get_group_id(int dimension) const { return m_group_item.get_id()[dimension]; }

    [[deprecated("deprecated in SYCL2020")]] range<Dimensions> get_global_range() const {
        return m_global_item.get_range();
    }

    size_t get_global_range(int dimension) const { return get_global_range()[dimension]; }

    id_type get_local_id() const { return m_local_item.get_id(); }

    size_t get_local_id(int dimension) const { return get_local_id()[dimension]; }

    size_t get_local_linear_id() const { return m_local_item.get_linear_id(); }

    range_type get_local_range() const { return m_local_item.get_range(); }

    size_t get_local_range(int dimension) const { return get_local_range()[dimension]; }

    size_t get_local_linear_range() const { return get_local_range().size(); }

    range_type get_group_range() const { return m_group_item.get_range(); }

    size_t get_group_range(int dimension) const { return get_group_range()[dimension]; }

    size_t get_group_linear_range() const { return m_group_item.size(); }

    range_type get_max_local_range() const { return get_local_range(); }

    size_t operator[](int dimension) const { return m_group_item.get_id()[dimension]; }

    [[deprecated("use sycl::group::get_group_linear_id")]] size_t get_linear_id() const {
        return get_group_linear_id();
    }

    size_t get_group_linear_id() const { return m_group_item.get_linear_id(); }

    bool leader() const { return (get_local_linear_id() == 0); }

    template <typename WorkItemFunctionT>
    void parallel_for_work_item(WorkItemFunctionT func) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(func);
    }

    template <typename WorkItemFunctionT>
    void parallel_for_work_item(range<Dimensions> flexible_range, WorkItemFunctionT func) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(flexible_range, func);
    }

    template <access::mode AccessMode = access_mode::read_write>
    void mem_fence(typename std::enable_if_t<AccessMode == access_mode::read || AccessMode == access_mode::write
                           || AccessMode == access_mode::read_write,
                       access::fence_space>
                       access_space
        = access::fence_space::global_and_local) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(access_space);
    }

    template <typename... Events>
    void wait_for(Events... events) const {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(events...);
    }

    bool operator==(const group<Dimensions> &rhs) const {
        return (rhs.m_local_id == this->m_local_id) && (rhs.m_local_range == this->m_local_range)
            && (rhs.m_global_id == this->m_global_id) && (rhs.m_global_range == this->m_global_range)
            && (rhs.m_group_id == this->m_group_id) && (rhs.m_group_range == this->m_group_range)
            && (rhs.m_impl == this->m_impl);
    }

    bool operator!=(const group<Dimensions> &rhs) const { return !((*this) == rhs); }

  private:
    item<Dimensions, false> m_local_item;
    item<Dimensions, false> m_global_item;
    item<Dimensions, false> m_group_item;
    detail::group_impl *m_impl;

    group(const item<Dimensions, false> &local_item, const item<Dimensions, false> &global_item,
        const item<Dimensions, false> &group_item, detail::group_impl *impl)
        : m_local_item(local_item), m_global_item(global_item), m_group_item(group_item), m_impl(impl) {}

    friend group<Dimensions> detail::make_group<Dimensions>(const sycl::item<Dimensions, false> &local_item,
        const sycl::item<Dimensions, false> &global_item, const sycl::item<Dimensions, false> &group_item,
        detail::group_impl *impl);

    friend detail::group_impl &detail::get_group_impl<Dimensions>(sycl::group<Dimensions> &g);
};

template <int Dimensions>
struct is_group<group<Dimensions>> : std::true_type {};

} // namespace simsycl::sycl
