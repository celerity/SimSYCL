#pragma once

#include "forward.hh"
#include "simsycl/detail/coordinate.hh"

namespace simsycl::sycl {

template<int Dimensions>
class range : public detail::coordinate<range<Dimensions>, Dimensions> {
  public:
    constexpr range() {
        for(int d = 0; d < Dimensions; ++d) { (*this)[d] = 0; }
    }

    template<typename... Values>
        requires(sizeof...(Values) == Dimensions)
    constexpr range(const Values... dims) : detail::coordinate<range<Dimensions>, Dimensions>(dims...) {}

    constexpr size_t size() const {
        size_t s = 1;
        for(int d = 0; d < Dimensions; ++d) { s *= (*this)[d]; }
        return s;
    }

  private:
    friend class detail::coordinate<range<Dimensions>, Dimensions>;
};

range(size_t) -> range<1>;
range(size_t, size_t) -> range<2>;
range(size_t, size_t, size_t) -> range<3>;

} // namespace simsycl::sycl
