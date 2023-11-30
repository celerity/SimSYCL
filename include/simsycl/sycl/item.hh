#pragma once

#include "forward.hh"
#include "id.hh"

#include "range.hh"
#include <type_traits>

namespace simsycl::detail {

template <int Dimensions>
sycl::item<Dimensions, true> make_item(
    const sycl::id<Dimensions> &the_id, const sycl::range<Dimensions> &range, const sycl::id<Dimensions> &offset) {
    return sycl::item<Dimensions, true>(the_id, range, offset);
}
template <int Dimensions>
sycl::item<Dimensions, false> make_item(const sycl::id<Dimensions> &the_id, const sycl::range<Dimensions> &range) {
    return sycl::item<Dimensions, false>(the_id, range, sycl::id<Dimensions>::zero());
}

} // namespace simsycl::detail

namespace simsycl::sycl {

template <int Dimensions, bool WithOffset>
class item {
  public:
    static_assert(Dimensions >= 1 && Dimensions <= 3);

    static constexpr int dimensions = Dimensions;

    id<Dimensions> get_id() const { return m_id; }

    size_t get_id(int dimension) const { return m_id[dimension]; }

    size_t operator[](int dimension) const { return m_id[dimension]; }

    range<Dimensions> get_range() const { return m_range; }

    size_t get_range(int dimension) const { return m_range[dimension]; }

    [[deprecated("Deprecated in SYCL 2020")]] id<Dimensions> get_offset() const
        requires WithOffset
    {
        return m_offset;
    }

    operator item<Dimensions, true>() const
        requires(!WithOffset)
    {
        return item<Dimensions, true>(m_id, m_range, m_offset);
    }

    operator size_t() const
        requires(Dimensions == 1)
    {
        return m_id;
    }

    size_t get_linear_id() const {
        size_t linear = m_id[0];
        for(int d = 1; d < Dimensions; ++d) { linear = linear * m_range[d] + m_id[d] - m_offset[d]; }
        return linear;
    }

  private:
    friend sycl::item<Dimensions, true> simsycl::detail::make_item<Dimensions>(
        const sycl::id<Dimensions> &, const sycl::range<Dimensions> &, const sycl::id<Dimensions> &);
    friend sycl::item<Dimensions, false> simsycl::detail::make_item<Dimensions>(
        const sycl::id<Dimensions> &, const sycl::range<Dimensions> &);

    id<Dimensions> m_id;
    range<Dimensions> m_range;
    id<Dimensions> m_offset;

    item(const id<Dimensions> &the_id, const range<Dimensions> &range, const id<Dimensions> &offset)
        : m_id(the_id), m_range(range), m_offset(offset) {}
};

} // namespace simsycl::sycl