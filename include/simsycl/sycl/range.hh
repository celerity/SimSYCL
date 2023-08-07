#pragma once

#include "forward.hh"
#include "../detail/coordinate.hh"

namespace simsycl::sycl {

template <int Dimensions>
class range : public detail::coordinate<range<Dimensions>, Dimensions> {
  public:
    template <typename... Values, typename = std::enable_if_t<sizeof...(Values) + 1 == Dimensions>>
    constexpr range(const size_t dim_0, const Values... dim_n)
        : detail::coordinate<range<Dimensions>, Dimensions>(dim_0, dim_n...) {}

    constexpr size_t size() const {
        size_t s = 1;
        for(int d = 0; d < Dimensions; ++d) { s *= (*this)[d]; }
        return s;
    }

  private:
    friend class detail::coordinate<range<Dimensions>, Dimensions>;

    constexpr range() noexcept {}
};

range(size_t)->range<1>;
range(size_t, size_t)->range<2>;
range(size_t, size_t, size_t)->range<3>;

} // namespace simsycl::sycl
