#pragma once

#include "forward.hh"
#include "id.hh"
#include "range.hh"

namespace simsycl::sycl {

template <int Dimensions>
class nd_range {
  public:
    static constexpr int dimensions = Dimensions;

    nd_range(range<Dimensions> global_range, range<Dimensions> local_range)
        : m_global_range(global_range), m_local_range(local_range) {}

    [[deprecated("Deprecated in SYCL 2020")]] nd_range(
        range<Dimensions> global_range, range<Dimensions> local_range, id<Dimensions> offset);

    range<Dimensions> get_global_range() const { return m_global_range; }
    range<Dimensions> get_local_range() const { return m_local_range; }
    range<Dimensions> get_group_range() const { return m_global_range / m_local_range; }
    [[deprecated("Deprecated in SYCL 2020")]] id<Dimensions> get_offset() const { return m_offset; }

    friend bool operator==(const nd_range &lhs, const nd_range &rhs) {
        return lhs.m_global_range == rhs.m_global_range && lhs.m_local_range == rhs.m_local_range
            && lhs.m_offset == rhs.m_offset;
    }

    friend bool operator!=(const nd_range &lhs, const nd_range &rhs) { return !(lhs == rhs); }

  private:
    range<Dimensions> m_global_range;
    range<Dimensions> m_local_range;
    id<Dimensions> m_offset;
};

} // namespace simsycl::sycl
