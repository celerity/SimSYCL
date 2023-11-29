#pragma once

namespace simsycl::detail {

template <typename T, typename T2>
auto div_ceil(T a, T2 b) {
    return (a + b - 1) / b;
}

} // namespace simsycl::detail
