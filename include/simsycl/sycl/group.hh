#pragma once

#include "forward.hh"

#include "h_item.hh"
#include "id.hh"
#include "item.hh"
#include "multi_ptr.hh"
#include "range.hh"
#include "type_traits.hh"

#include "simsycl/detail/check.hh"
#include "simsycl/detail/group_operation_impl.hh"

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

template<typename G, int Dimensions>
class hierarchical_group_size_setter {
  public:
    hierarchical_group_size_setter(G &g, sycl::range<Dimensions> flexible_size)
        : m_g(g), m_old_local_item(g.m_local_item), m_old_global_item(g.m_global_item) {
        g.m_local_item = simsycl::detail::make_item(sycl::id<Dimensions>(), flexible_size);
        g.m_global_item
            = simsycl::detail::make_item(sycl::id<Dimensions>(), g.m_group_item.get_range() * flexible_size);
    }

    ~hierarchical_group_size_setter() {
        m_g.m_local_item = m_old_local_item;
        m_g.m_global_item = m_old_global_item;
    }

  private:
    G &m_g;
    sycl::item<Dimensions, false> m_old_local_item;
    sycl::item<Dimensions, true> m_old_global_item;
};

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
        return m_local_item.get_id();
    }

    size_t get_local_id(int dimension) const { return get_local_id()[dimension]; }

    size_t get_local_linear_id() const {
        SIMSYCL_CHECK(m_type == detail::group_type::nd_range
            && "get_local_linear_id is not supported for from within a parallel_for_work_item context");
        return m_local_item.get_linear_id();
    }

    range_type get_local_range() const { return m_local_item.get_range(); }

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
        parallel_for_work_item(m_local_item.get_range(), func);
    }

    // All parallel_for_work_item calls within a given parallel_for_work_group execution must have the same dimensions
    template<typename WorkItemFunctionT>
    void parallel_for_work_item(range<Dimensions> flexible_range, WorkItemFunctionT func) const {
        SIMSYCL_CHECK(m_type != detail::group_type::nd_range
            && "parallel_for_work_item is only supported for from within a parallel_for_work_item context");
        // TODO get rid of hierarchical_group_size_setter and mutable members, just create a second group instance
        detail::hierarchical_group_size_setter set(*this, flexible_range);
        if constexpr(Dimensions == 1) {
            for(size_t i = 0; i < flexible_range[0]; ++i) {
                const auto global_id = m_group_item.get_id() * flexible_range[0] + i;
                const auto global_range = m_group_item.get_range() * flexible_range[0];
                const auto local_id = id<1>(global_id[0] % flexible_range[0]);
                const auto global_item = simsycl::detail::make_item(global_id, global_range);
                const auto local_item = simsycl::detail::make_item(local_id, flexible_range);
                func(detail::make_h_item(global_item, local_item, local_item));
            }
        } else if constexpr(Dimensions == 2) {
            for(size_t i = 0; i < flexible_range[0]; ++i) {
                for(size_t j = 0; j < flexible_range[1]; ++j) {
                    const auto global_id = id<2>(
                        m_group_item.get_id(0) * flexible_range[0] + i, m_group_item.get_id(1) * flexible_range[1] + j);
                    const auto global_range = range<2>(
                        m_group_item.get_range(0) * flexible_range[0], m_group_item.get_range(1) * flexible_range[1]);
                    const auto local_id = id<2>(global_id[0] % flexible_range[0], global_id[1] % flexible_range[1]);
                    const auto global_item = simsycl::detail::make_item(global_id, global_range);
                    const auto local_item = simsycl::detail::make_item(local_id, flexible_range);
                    func(detail::make_h_item(global_item, local_item, local_item));
                }
            }
        } else if constexpr(Dimensions == 3) {
            for(size_t i = 0; i < flexible_range[0]; ++i) {
                for(size_t j = 0; j < flexible_range[1]; ++j) {
                    for(size_t k = 0; k < flexible_range[2]; ++k) {
                        const auto global_id = id<3>(m_group_item.get_id(0) * flexible_range[0] + i,
                            m_group_item.get_id(1) * flexible_range[1] + j,
                            m_group_item.get_id(2) * flexible_range[2] + k);
                        const auto global_range = range<3>(m_group_item.get_range(0) * flexible_range[0],
                            m_group_item.get_range(1) * flexible_range[1],
                            m_group_item.get_range(2) * flexible_range[2]);
                        const auto local_id = id<3>(global_id[0] % flexible_range[0], global_id[1] % flexible_range[1],
                            global_id[2] % flexible_range[2]);
                        const auto global_item = simsycl::detail::make_item(global_id, global_range);
                        const auto local_item = simsycl::detail::make_item(local_id, flexible_range);
                        func(detail::make_h_item(global_item, local_item, local_item));
                    }
                }
            }
        }
    }

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
        return lhs.m_local_item == rhs.m_local_item && lhs.m_global_item == rhs.m_global_item
            && lhs.m_group_item == rhs.m_group_item && lhs.m_concurrent_group == rhs.m_concurrent_group;
    }

    friend bool operator!=(const group<Dimensions> &lhs, const group<Dimensions> &rhs) { return !(lhs == rhs); }

  private:
    template<typename G, int D>
    friend class detail::hierarchical_group_size_setter;

    friend group<Dimensions> detail::make_group<Dimensions>(const detail::group_type type,
        const sycl::item<Dimensions, false> &local_item, const sycl::item<Dimensions, true> &global_item,
        const sycl::item<Dimensions, false> &group_item, detail::concurrent_group *impl);

    friend detail::group_type detail::get_group_type<Dimensions>(const sycl::group<Dimensions> &g);
    friend detail::concurrent_group &detail::get_concurrent_group<Dimensions>(const sycl::group<Dimensions> &g);

    detail::group_type m_type;
    mutable item<Dimensions, false /* WithOffset */> m_local_item; // mutable for hierarchical_group_size_setter
    mutable item<Dimensions, true /* WithOffset */> m_global_item; // mutable for hierarchical_group_size_setter
    item<Dimensions, false /* WithOffset */> m_group_item;
    detail::concurrent_group *m_concurrent_group;

    group(const detail::group_type type, const item<Dimensions, false> &local_item,
        const item<Dimensions, true> &global_item, const item<Dimensions, false> &group_item,
        detail::concurrent_group *impl)
        : m_type(type), m_local_item(local_item), m_global_item(global_item), m_group_item(group_item),
          m_concurrent_group(impl) {}
};

template<int Dimensions>
struct is_group<group<Dimensions>> : std::true_type {};

} // namespace simsycl::sycl
