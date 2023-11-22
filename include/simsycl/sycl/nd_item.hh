#pragma once

#include "forward.hh"

#include "group.hh"
#include "id.hh"
#include "range.hh"
#include "simsycl/detail/check.hh"
#include "sub_group.hh"

namespace simsycl::sycl {

template <int Dimensions = 1>
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

    size_t get_group(int dimension) const {return m_group.get_group_id(dimension); }

    size_t get_group_linear_id() const {
        return m_group.get_group_linear_id();
    }

    range<Dimensions> get_group_range() const { return m_group.get_group_range(); }

    size_t get_group_range(int dimension) const { return m_group.get_group_range(dimension); }

    range<Dimensions> get_global_range() const { return m_global_item.get_range(); }

    size_t get_global_range(int dimension) const { return m_global_item.get_range(dimension); }

    range<Dimensions> get_local_range() const { return m_local_item.get_range(); }

    size_t get_local_range(int dimension) const { return m_local_item.get_range(dimension); }

    [[deprecated("offsets are deprecated in SYCL 2020")]] id<Dimensions> get_offset() const {
        return m_global_item.get_offset();
    }

    nd_range<Dimensions> get_nd_range() const {
        return nd_range<Dimensions>(get_global_range(), get_local_range(), get_offset());
    }

    void barrier(access::fence_space access_space = access::fence_space::global_and_local) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <access::mode AccessMode = access_mode::read_write>
    [[deprecated("use sycl::atomic_fence() free function instead")]] void mem_fence(
        typename std::enable_if_t<AccessMode == access_mode::read || AccessMode == access_mode::write
                || AccessMode == access_mode::read_write,
            access::fence_space>
            access_space
        = access::fence_space::global_and_local) const {
        SIMSYCL_NOT_IMPLEMENTED;
    }

    template <typename... Events>
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
    item<Dimensions, false> m_global_item;
    item<Dimensions, false> m_local_item;
    group<Dimensions> m_group;
    sub_group m_sub_group;
};

} // namespace simsycl::sycl
