#pragma once

#include "forward.hh"

#include "h_item.hh"
#include "id.hh"
#include "item.hh"
#include "multi_ptr.hh"
#include "range.hh"
#include "type_traits.hh"

#include "../detail/check.hh"
#include "../detail/group_operation_impl.hh"
#include "../detail/utils.hh"


namespace simsycl::detail {

enum class group_type { nd_range, hierarchical_implicit_size, hierarchical_explicit_size };

template<int Dimensions>
sycl::group<Dimensions> make_group(const group_type type, const sycl::item<Dimensions, false> &local_item,
    const sycl::item<Dimensions, true> &global_item, const sycl::item<Dimensions, false> &group_item,
    detail::concurrent_group *impl) {
    return sycl::group<Dimensions>(type, local_item, global_item, group_item, impl);
}

template<int Dimensions>
group_type get_group_type(const sycl::group<Dimensions> &g) {
    return g.m_type;
}

} // namespace simsycl::detail

namespace simsycl::sycl {

class device_event {
  public:
    void wait() noexcept {}
};

template<int Dimensions>
class group {
  public:
    using id_type = id<Dimensions>;
    using range_type = range<Dimensions>;
    using linear_id_type = size_t;
    static constexpr int dimensions = Dimensions;
    static constexpr sycl::memory_scope fence_scope = sycl::memory_scope::work_group;

    group() = delete;

    // TODO not in the spec, remove
    [[deprecated("non-standard, use sycl::group::get_group_id")]] id_type get_id() const { return get_group_id(); }

    // TODO not in the spec, remove
    [[deprecated("non-standard, use sycl::group::get_group_id")]] size_t get_id(int dimension) const {
        return get_group_id(dimension);
    }

    id_type get_group_id() const { return m_group_item.get_id(); }

    size_t get_group_id(int dimension) const { return m_group_item.get_id()[dimension]; }

    [[deprecated("non-standard")]] range<Dimensions> get_global_range() const {
        SIMSYCL_CHECK(
            m_global_item.get_range().size() != 0 && "get_global_range called from hierarchical group scope?");
        return m_global_item.get_range();
    }

    [[deprecated("non-standard")]] size_t get_global_range(int dimension) const {
        return get_global_range()[dimension];
    }

    id_type get_local_id() const {
        SIMSYCL_CHECK(m_type == detail::group_type::nd_range
            && "get_local_id is not supported for from within a parallel_for_work_item context");
        return m_physical_local_item.get_id();
    }

    size_t get_local_id(int dimension) const { return get_local_id()[dimension]; }

    size_t get_local_linear_id() const {
        SIMSYCL_CHECK(m_type == detail::group_type::nd_range
            && "get_local_linear_id is not supported for from within a parallel_for_work_item context");
        return m_physical_local_item.get_linear_id();
    }

    range_type get_local_range() const { return m_physical_local_item.get_range(); }

    size_t get_local_range(int dimension) const { return get_local_range()[dimension]; }

    size_t get_local_linear_range() const { return get_local_range().size(); }

    range_type get_group_range() const { return m_group_item.get_range(); }

    size_t get_group_range(int dimension) const { return get_group_range()[dimension]; }

    size_t get_group_linear_range() const { return get_group_range().size(); }

    range_type get_max_local_range() const { return get_local_range(); }

    size_t operator[](int dimension) const { return m_group_item.get_id()[dimension]; }

    [[deprecated("non-standard, use sycl::group::get_group_linear_id")]] size_t get_linear_id() const {
        return get_group_linear_id();
    }

    size_t get_group_linear_id() const { return m_group_item.get_linear_id(); }

    bool leader() const {
        SIMSYCL_CHECK(m_type == detail::group_type::nd_range
            && "leader() is not supported for from within a parallel_for_work_item context");
        return (get_local_linear_id() == 0);
    }

    template<typename WorkItemFunctionT>
    void parallel_for_work_item(WorkItemFunctionT func) const {
        SIMSYCL_CHECK(m_type != detail::group_type::nd_range
            && "parallel_for_work_item is only supported for from within a parallel_for_work_item context");
        SIMSYCL_CHECK(m_type != detail::group_type::hierarchical_implicit_size
            && "parallel_for_work_item(func) without a range argument is only supported in a parallel_for_work_item "
               "context with a set local range");
        parallel_for_work_item(m_physical_local_item.get_range(), func);
    }

    SIMSYCL_START_IGNORING_DEPRECATIONS

    // All parallel_for_work_item calls within a given parallel_for_work_group execution must have the same dimensions
    template<typename WorkItemFunctionT>
    void parallel_for_work_item(range<Dimensions> flexible_range, WorkItemFunctionT func) const {
        SIMSYCL_CHECK(m_type != detail::group_type::nd_range
            && "parallel_for_work_item is only supported for from within a parallel_for_work_item context");

        SIMSYCL_CHECK(m_global_item.get_offset() == sycl::id<Dimensions>{});

        detail::for_each_id_in_range(flexible_range, [&](const id<Dimensions> &logical_local_id) {
            const auto logical_local_item = simsycl::detail::make_item(logical_local_id, flexible_range);
            const auto physical_local_item = simsycl::detail::make_item(
                logical_local_id % id(m_physical_local_item.get_range()), m_physical_local_item.get_range());
            const auto global_item
                = detail::make_item(m_global_item.get_id() + physical_local_item.get_id(), m_global_item.get_range());
            func(detail::make_h_item(global_item, logical_local_item, physical_local_item));
        });
    }

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

    // TODO not in the spec, remove
    template<access_mode AccessMode = access_mode::read_write>
    void mem_fence(typename std::enable_if_t<AccessMode == access_mode::read || AccessMode == access_mode::write
                           || AccessMode == access_mode::read_write,
                       access::fence_space>
                       access_space
        = access::fence_space::global_and_local) const {
        (void)(access_space);
        // mem_fence is a no-op in SimSYCL
    }

    template<typename... Events>
    void wait_for(Events... events) const {
        simsycl::detail::sink{events...};
        // wait_for is a no-op in SimSYCL
    }

    template<typename DataT>
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL device_event async_work_group_copy(
        local_ptr<DataT> dest, global_ptr<DataT> src, size_t num_elements) const {
        std::copy_n(src.get(), num_elements, dest.get());
        return device_event{};
    }

    template<typename DataT>
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL device_event async_work_group_copy(
        global_ptr<DataT> dest, local_ptr<DataT> src, size_t num_elements) const {
        std::copy_n(src.get(), num_elements, dest.get());
        return device_event{};
    }

    template<typename DataT>
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL device_event async_work_group_copy(
        local_ptr<DataT> dest, global_ptr<DataT> src, size_t num_elements, size_t src_stride) const {
        for(size_t i = 0; i < num_elements; ++i) { dest[i] = src[i * src_stride]; }
        return device_event{};
    }

    template<typename DataT>
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL device_event async_work_group_copy(
        global_ptr<DataT> dest, local_ptr<DataT> src, size_t num_elements, size_t dest_stride) const {
        for(size_t i = 0; i < num_elements; ++i) { dest[i * dest_stride] = src[i]; }
        return device_event{};
    }

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(
        decorated_local_ptr<DestDataT> dest, decorated_global_ptr<SrcDataT> src, size_t num_elements) const {
        std::copy_n(src.get(), num_elements, dest.get());
        return device_event{};
    }

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(
        decorated_global_ptr<DestDataT> dest, decorated_local_ptr<SrcDataT> src, size_t num_elements) const {
        std::copy_n(src.get(), num_elements, dest.get());
        return device_event{};
    }

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(decorated_local_ptr<DestDataT> dest, decorated_global_ptr<SrcDataT> src,
        size_t num_elements, size_t src_stride) const {
        for(size_t i = 0; i < num_elements; ++i) { dest[i] = src[i * src_stride]; }
        return device_event{};
    }

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(decorated_global_ptr<DestDataT> dest, decorated_local_ptr<SrcDataT> src,
        size_t num_elements, size_t dest_stride) const {
        for(size_t i = 0; i < num_elements; ++i) { dest[i * dest_stride] = src[i]; }
        return device_event{};
    }

    friend bool operator==(const group<Dimensions> &lhs, const group<Dimensions> &rhs) {
        return lhs.m_physical_local_item == rhs.m_physical_local_item && lhs.m_global_item == rhs.m_global_item
            && lhs.m_group_item == rhs.m_group_item && lhs.m_concurrent_group == rhs.m_concurrent_group;
    }

    friend bool operator!=(const group<Dimensions> &lhs, const group<Dimensions> &rhs) { return !(lhs == rhs); }

  private:
    friend group<Dimensions> detail::make_group<Dimensions>(const detail::group_type type,
        const sycl::item<Dimensions, false> &local_item, const sycl::item<Dimensions, true> &global_item,
        const sycl::item<Dimensions, false> &group_item, detail::concurrent_group *impl);

    friend detail::group_type detail::get_group_type<Dimensions>(const sycl::group<Dimensions> &g);
    friend detail::concurrent_group &detail::get_concurrent_group<Dimensions>(const sycl::group<Dimensions> &g);

    detail::group_type m_type;
    item<Dimensions, false /* WithOffset */> m_physical_local_item;
    item<Dimensions, true /* WithOffset */> m_global_item;
    item<Dimensions, false /* WithOffset */> m_group_item;
    detail::concurrent_group *m_concurrent_group = nullptr;

    group(const detail::group_type type, const item<Dimensions, false> &physical_local_item,
        const item<Dimensions, true> &global_item, const item<Dimensions, false> &group_item,
        detail::concurrent_group *concurrent_group)
        : m_type(type), m_physical_local_item(physical_local_item), m_global_item(global_item),
          m_group_item(group_item), m_concurrent_group(concurrent_group) {}
};

template<int Dimensions>
struct is_group<group<Dimensions>> : std::true_type {};

} // namespace simsycl::sycl
