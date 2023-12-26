#pragma once

#include "forward.hh"

#include "group.hh"
#include "id.hh"
#include "multi_ptr.hh"
#include "range.hh"
#include "sub_group.hh"

#include "simsycl/detail/check.hh"


namespace simsycl::detail {

struct nd_item_instance {
    size_t group_ops_reached = 0;
    size_t sub_group_ops_reached = 0;
};

struct concurrent_nd_item {
  public:
    detail::concurrent_group *concurrent_group = nullptr;
    nd_item_instance instance;
};

template<int Dimensions>
sycl::nd_item<Dimensions> make_nd_item(const sycl::item<Dimensions, true> &global_id,
    const sycl::item<Dimensions, false> &local_id, const sycl::group<Dimensions> &group,
    const sycl::sub_group &sub_group, concurrent_nd_item *impl) {
    return sycl::nd_item<Dimensions>(global_id, local_id, group, sub_group, impl);
}

} // namespace simsycl::detail

namespace simsycl::sycl {

class device_event {
  public:
    void wait() noexcept {}
};

template<int Dimensions>
class nd_item {
  public:
    static constexpr int dimensions = Dimensions;

    nd_item() = delete;

    id<Dimensions> get_global_id() const { return m_global_item.get_id(); }

    size_t get_global_id(int dimension) const { return m_global_item.get_id(dimension); }

    size_t get_global_linear_id() const { return m_global_item.get_linear_id(); }

    id<Dimensions> get_local_id() const { return m_local_item.get_id(); }

    size_t get_local_id(int dimension) const { return m_local_item.get_id(dimension); }

    size_t get_local_linear_id() const { return m_local_item.get_linear_id(); }

    group<Dimensions> get_group() const { return m_group; }

    sub_group get_sub_group() const { return m_sub_group; }

    size_t get_group(int dimension) const { return m_group.get_group_id(dimension); }

    size_t get_group_linear_id() const { return m_group.get_group_linear_id(); }

    range<Dimensions> get_group_range() const { return m_group.get_group_range(); }

    size_t get_group_range(int dimension) const { return m_group.get_group_range(dimension); }

    range<Dimensions> get_global_range() const { return m_global_item.get_range(); }

    size_t get_global_range(int dimension) const { return m_global_item.get_range(dimension); }

    range<Dimensions> get_local_range() const { return m_local_item.get_range(); }

    size_t get_local_range(int dimension) const { return m_local_item.get_range(dimension); }

    nd_range<Dimensions> get_nd_range() const {
        return nd_range<Dimensions>(get_global_range(), get_local_range(), m_global_item.get_offset());
    }

    SIMSYCL_START_IGNORING_DEPRECATIONS

    [[deprecated("offsets are deprecated in SYCL 2020")]] id<Dimensions> get_offset() const {
        return m_global_item.get_offset();
    }

    [[deprecated("use sycl::group_barrier() free function instead")]] void barrier(
        access::fence_space access_space = access::fence_space::global_and_local) const {
        (void)access_space;
        detail::yield_to_kernel_scheduler();
    }

    template<access::mode AccessMode = access_mode::read_write>
    [[deprecated("use sycl::atomic_fence() free function instead")]] void mem_fence(
        typename std::enable_if_t<AccessMode == access_mode::read || AccessMode == access_mode::write
                || AccessMode == access_mode::read_write,
            access::fence_space>
            access_space
        = access::fence_space::global_and_local) const {
        SIMSYCL_NOT_IMPLEMENTED(access_space);
    }

    template<typename DataT>
    [[deprecated]] device_event async_work_group_copy(
        local_ptr<DataT> dest, global_ptr<DataT> src, size_t num_elements) const;

    template<typename DataT>
    [[deprecated]] device_event async_work_group_copy(
        global_ptr<DataT> dest, local_ptr<DataT> src, size_t num_elements) const;

    template<typename DataT>
    [[deprecated]] device_event async_work_group_copy(
        local_ptr<DataT> dest, global_ptr<DataT> src, size_t num_elements, size_t src_stride) const;

    template<typename DataT>
    [[deprecated]] device_event async_work_group_copy(
        global_ptr<DataT> dest, local_ptr<DataT> src, size_t num_elements, size_t dest_stride) const;

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(
        decorated_local_ptr<DestDataT> dest, decorated_global_ptr<SrcDataT> src, size_t num_elements) const;

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(
        decorated_global_ptr<DestDataT> dest, decorated_local_ptr<SrcDataT> src, size_t num_elements) const;

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(decorated_local_ptr<DestDataT> dest, decorated_global_ptr<SrcDataT> src,
        size_t num_elements, size_t src_stride) const;

    template<typename DestDataT, typename SrcDataT>
        requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
    device_event async_work_group_copy(decorated_global_ptr<DestDataT> dest, decorated_local_ptr<SrcDataT> src,
        size_t num_elements, size_t dest_stride) const;

    template<typename... Events>
    void wait_for(Events... events) const {
        m_group.wait_for(events...);
    }

    nd_item(const nd_item &rhs) = default;

    nd_item(nd_item &&rhs) = default;

    nd_item &operator=(const nd_item &rhs) = default;

    nd_item &operator=(nd_item &&rhs) = default;

    bool operator==(const nd_item &rhs) const {
        return (rhs.m_local_item == this->m_local_item) && (rhs.m_global_item == this->m_global_item)
            && (rhs.m_group == this->m_group);
    }

    bool operator!=(const nd_item &rhs) const { return !((*this) == rhs); }

  private:
    item<Dimensions, true /* WithOffset */> m_global_item;
    item<Dimensions, false /* WithOffset */> m_local_item;
    group<Dimensions> m_group;
    sub_group m_sub_group;

    detail::concurrent_nd_item *m_impl;

    nd_item(const item<Dimensions, true> &global_item, const item<Dimensions, false> &local_item,
        const group<Dimensions> &group, const sub_group &sub_group, detail::concurrent_nd_item *impl)
        : m_global_item(global_item), m_local_item(local_item), m_group(group), m_sub_group(sub_group), m_impl(impl) {}

    friend nd_item<Dimensions> detail::make_nd_item<Dimensions>(const sycl::item<Dimensions, true> &,
        const sycl::item<Dimensions, false> &, const sycl::group<Dimensions> &, const sycl::sub_group &,
        detail::concurrent_nd_item *);
};

} // namespace simsycl::sycl
