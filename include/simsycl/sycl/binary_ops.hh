#pragma once

#include "concepts.hh"
#include "type_traits.hh"

#include <functional>

namespace simsycl::sycl {

// arithmetic

using std::plus;
template<typename OpT, typename T>
struct has_known_identity<plus<OpT>, T> : std::bool_constant<detail::is_arithmetic_v<T>> {};
template<typename OpT, Arithmetic T>
struct known_identity<plus<OpT>, T> {
    static constexpr std::remove_cv_t<T> value = T{};
};

using std::multiplies;
template<typename OpT, typename T>
struct has_known_identity<multiplies<OpT>, T> : std::bool_constant<detail::is_arithmetic_v<T>> {};
template<typename OpT, Arithmetic T>
struct known_identity<multiplies<OpT>, T> {
    static constexpr std::remove_cv_t<T> value = T{1};
};

// bitwise boolean logic

using std::bit_and;
template<typename OpT, typename T>
struct has_known_identity<bit_and<OpT>, T> : std::bool_constant<std::is_integral_v<T>> {};
template<typename OpT, std::integral T>
struct known_identity<bit_and<OpT>, T> {
    static constexpr std::remove_cv_t<T> value = ~T{};
};

using std::bit_or;
template<typename OpT, typename T>
struct has_known_identity<bit_or<OpT>, T> : std::bool_constant<std::is_integral_v<T>> {};
template<typename OpT, std::integral T>
struct known_identity<bit_or<OpT>, T> {
    static constexpr std::remove_cv_t<T> value = T{};
};

using std::bit_xor;
template<typename OpT, typename T>
struct has_known_identity<bit_xor<OpT>, T> : std::bool_constant<std::is_integral_v<T>> {};
template<typename OpT, std::integral T>
struct known_identity<bit_xor<OpT>, T> {
    static constexpr std::remove_cv_t<T> value = T{};
};

// logic

using std::logical_and;
template<typename OpT, typename T>
struct has_known_identity<logical_and<OpT>, T> : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};
template<typename OpT, Boolean T>
struct known_identity<logical_and<OpT>, T> {
    static constexpr std::remove_cv_t<T> value = true;
};

using std::logical_or;
template<typename OpT, typename T>
struct has_known_identity<logical_or<OpT>, T> : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};
template<typename OpT, Boolean T>
struct known_identity<logical_or<OpT>, T> {
    static constexpr std::remove_cv_t<T> value = false;
};

// min/max

template<typename T = void>
struct minimum {
    T operator()(const T &x, const T &y) const { return x < y ? x : y; }
};
template<>
struct minimum<void> {
    template<typename T, typename U>
    std::common_type_t<T &&, U &&> operator()(T &&x, U &&y) const {
        return x < y ? std::forward<T>(x) : std::forward<U>(y);
    }
};
template<typename OpT, typename T>
struct has_known_identity<minimum<OpT>, T> : std::bool_constant<detail::is_arithmetic_v<T>> {};
template<typename OpT, Arithmetic T>
struct known_identity<minimum<OpT>, T> {
    static constexpr std::remove_cv_t<T> value
        = std::is_floating_point_v<T> ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
};

template<typename T = void>
struct maximum {
    T operator()(const T &x, const T &y) const { return x >= y ? x : y; }
};
template<>
struct maximum<void> {
    template<typename T, typename U>
    std::common_type_t<T &&, U &&> operator()(T &&x, U &&y) const {
        return x > y ? std::forward<T>(x) : std::forward<U>(y);
    }
};
template<typename OpT, typename T>
struct has_known_identity<maximum<OpT>, T> : std::bool_constant<detail::is_arithmetic_v<T>> {};
template<typename OpT, Arithmetic T>
struct known_identity<maximum<OpT>, T> {
    static constexpr std::remove_cv_t<T> value
        = std::is_floating_point_v<T> ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
};

} // namespace simsycl::sycl

namespace simsycl::detail {

template<typename T>
struct is_function_object<simsycl::sycl::plus<T>> : std::true_type {};
template<typename T>
struct is_function_object<simsycl::sycl::multiplies<T>> : std::true_type {};

template<typename T>
struct is_function_object<simsycl::sycl::bit_and<T>> : std::true_type {};
template<typename T>
struct is_function_object<simsycl::sycl::bit_or<T>> : std::true_type {};
template<typename T>
struct is_function_object<simsycl::sycl::bit_xor<T>> : std::true_type {};

template<typename T>
struct is_function_object<simsycl::sycl::logical_and<T>> : std::true_type {};
template<typename T>
struct is_function_object<simsycl::sycl::logical_or<T>> : std::true_type {};

template<typename T>
struct is_function_object<simsycl::sycl::minimum<T>> : std::true_type {};
template<typename T>
struct is_function_object<simsycl::sycl::maximum<T>> : std::true_type {};

} // namespace simsycl::detail
