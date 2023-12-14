#pragma once

#include "forward.hh"
#include "simsycl/detail/coordinate.hh"

namespace simsycl::sycl {

template<int Dimensions>
class range : public detail::coordinate<range<Dimensions>, Dimensions> {
  private:
    using coordinate = detail::coordinate<range<Dimensions>, Dimensions>;

  public:
    constexpr range() {
        for(int d = 0; d < Dimensions; ++d) { (*this)[d] = 0; }
    }

    template<std::convertible_to<size_t>... Values>
        requires(sizeof...(Values) + 1 == Dimensions)
    constexpr range(const size_t dim_0, const Values... dims) : coordinate(dim_0, dims...) {}

    constexpr size_t size() const {
        size_t s = 1;
        for(int d = 0; d < Dimensions; ++d) { s *= (*this)[d]; }
        return s;
    }

  private:
    friend coordinate;
};

range(size_t) -> range<1>;
range(size_t, size_t) -> range<2>;
range(size_t, size_t, size_t) -> range<3>;

} // namespace simsycl::sycl
