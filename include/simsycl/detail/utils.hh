#pragma once

#include "../sycl/id.hh"
#include "../sycl/range.hh"

#include <utility>

namespace simsycl::detail {

template<typename T, typename T2>
constexpr auto div_ceil(T a, T2 b) {
    return (a + b - 1) / b;
}

template<typename T>
constexpr T &&max(T &&x) {
    return std::forward<T>(x);
}

template<typename T, typename... Ts>
constexpr decltype(auto) max(T &&x, Ts &&...ts) {
    decltype(auto) rhs = max(std::forward<Ts>(ts)...);
    return x < rhs ? std::forward<decltype(rhs)>(rhs) : std::forward<T>(x);
}

template<typename T>
constexpr T &&min(T &&x) {
    return std::forward<T>(x);
}

template<typename T, typename... Ts>
constexpr decltype(auto) min(T &&x, Ts &&...ts) {
    decltype(auto) rhs = min(std::forward<Ts>(ts)...);
    return x < rhs ? std::forward<T>(x) : std::forward<decltype(rhs)>(rhs);
}

inline size_t get_linear_index(const sycl::range<1> &range, const sycl::id<1> &index) {
    SIMSYCL_CHECK(index[0] < range[0]);
    return index[0];
}

inline size_t get_linear_index(const sycl::range<2> &range, const sycl::id<2> &index) {
    SIMSYCL_CHECK(index[0] < range[0] && index[1] < range[1]);
    return index[0] * range[1] + index[1];
}

inline size_t get_linear_index(const sycl::range<3> &range, const sycl::id<3> &index) {
    SIMSYCL_CHECK(index[0] < range[0] && index[1] < range[1] && index[2] < range[2]);
    return index[0] * range[1] * range[2] + index[1] * range[2] + index[2];
}

template<int Dimensions>
sycl::id<Dimensions> linear_index_to_id(const sycl::range<Dimensions> &range, size_t linear_index) {
    sycl::id<Dimensions> id;
    for(int d = Dimensions - 1; d >= 0; --d) {
        id[d] = linear_index % range[d];
        linear_index /= range[d];
    }
    return id;
}

} // namespace simsycl::detail
