#pragma once

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

} // namespace simsycl::detail
