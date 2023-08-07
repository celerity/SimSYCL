#pragma once

#include "forward.hh"
#include "id.hh"

#include "range.hh"
#include <type_traits>

namespace simsycl::detail {

template <int Dimensions, bool WithOffset = true>
sycl::item<Dimensions, WithOffset> make_item(
    const sycl::id<Dimensions> &the_id, const sycl::range<Dimensions> &range, const sycl::id<Dimensions> &offset) {
    return sycl::item<Dimensions, WithOffset>(the_id, range, offset);
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

    template <bool W = WithOffset, std::enable_if_t<W, int> = 0>
    [[deprecated("Deprecated in SYCL 2020")]] id<Dimensions> get_offset() const {
        return m_offset;
    }

    template <bool W = WithOffset, std::enable_if_t<!W, int> = 0>
    operator item<Dimensions, true>() const {
        return item<Dimensions, true>(m_id, m_range, m_offset);
    }

    template <int D = Dimensions, std::enable_if_t<D == 1, int> = 0>
    operator size_t() const {
        return m_id;
    }

    size_t get_linear_id() const {
        size_t linear = m_id[0];
        for(int d = 1; d < Dimensions; ++d) { linear = linear * m_range[d] + m_id[d] - m_offset[d]; }
        return linear;
    }

  private:
    template <int D, bool W>
    friend sycl::item<D, W> make_item(
        const sycl::id<D> &the_id, const sycl::range<D> &range, const sycl::id<D> &offset);

    id<Dimensions> m_id;
    range<Dimensions> m_range;
    id<Dimensions> m_offset;

    item(const id<Dimensions> &the_id, const range<Dimensions> &range, const id<Dimensions> &offset)
        : m_id(the_id), m_range(range), m_offset(offset) {}
};

} // namespace simsycl::sycl