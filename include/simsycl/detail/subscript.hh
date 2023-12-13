#pragma once

#include "../sycl/id.hh"

#include <utility>

namespace simsycl::detail {

template<int TargetDimensions, typename Target, int SubscriptDimension = 0>
class subscript_proxy;

template<int TargetDimensions, typename Target, int SubscriptDimension>
inline decltype(auto) subscript(Target &tgt, sycl::id<TargetDimensions> id, const size_t index) {
    static_assert(SubscriptDimension < TargetDimensions);
    id[SubscriptDimension] = index;
    if constexpr(SubscriptDimension == TargetDimensions - 1) {
        return tgt[std::as_const(id)];
    } else {
        return subscript_proxy<TargetDimensions, Target, SubscriptDimension + 1>{tgt, id};
    }
}

template<int TargetDims, typename Target>
inline decltype(auto) subscript(Target &tgt, const size_t index) {
    return subscript<TargetDims, Target, 0>(tgt, sycl::id<TargetDims>{}, index);
}

template<int TargetDimensions, typename Target, int SubscriptDim>
class subscript_proxy {
  public:
    subscript_proxy(Target &tgt, const sycl::id<TargetDimensions> id) : m_tgt(tgt), m_id(id) {}

    inline decltype(auto) operator[](const size_t index) const {
        return subscript<TargetDimensions, Target, SubscriptDim>(m_tgt, m_id, index);
    }

  private:
    Target &m_tgt;
    sycl::id<TargetDimensions> m_id{};
};

} // namespace simsycl::detail
