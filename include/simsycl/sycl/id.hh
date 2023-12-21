#pragma once

#include "../detail/coordinate.hh"
#include "forward.hh"

namespace simsycl::sycl {

template<int Dimensions>
class id : public detail::coordinate<id<Dimensions>, Dimensions> {
  private:
    using coordinate = detail::coordinate<id<Dimensions>, Dimensions>;

  public:
    id() = default;

    template<std::convertible_to<size_t>... Values>
        requires(sizeof...(Values) + 1 == Dimensions)
    id(const size_t dim_0, const Values... dim_n) : coordinate(dim_0, dim_n...) {}

    id(const range<Dimensions> &range) {
        for(int d = 0; d < Dimensions; ++d) { (*this)[d] = range[d]; }
    }

    // Non-standard: The spec only mentions item<Dimensions> and thus only allows an item with offset to be passed here,
    // but we are certain that this is an oversight.
    template<bool WithOffset>
    id(const item<Dimensions, WithOffset> &item) {
        for(int d = 0; d < Dimensions; ++d) { (*this)[d] = item[d]; }
    }

    operator size_t() const
        requires(Dimensions == 1)
    {
        return (*this)[0];
    }

    static id<Dimensions> zero() {
        id<Dimensions> zero;
        for(int d = 0; d < Dimensions; ++d) { zero[d] = 0; }
        return zero;
    }
};

id(size_t) -> id<1>;
id(size_t, size_t) -> id<2>;
id(size_t, size_t, size_t) -> id<3>;

} // namespace simsycl::sycl
