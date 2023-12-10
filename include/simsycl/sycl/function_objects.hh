#pragma once

#include "concepts.hh"
#include "type_traits.hh"


namespace simsycl::sycl {

// arithmetic

template <typename T = void>
struct plus {
    T operator()(const T &x, const T &y) const { return x + y; }
};
template <typename T>
struct is_function_object<plus<T>> : std::true_type {};
template <typename T>
struct has_known_identity<plus<T>, T> : std::bool_constant<is_arithmetic_v<T>> {};
template <Arithmetic T>
struct known_identity<plus<T>, T> {
    static constexpr T value = T{};
};

template <typename T = void>
struct multiplies {
    T operator()(const T &x, const T &y) const { return x * y; }
};
template <typename T>
struct is_function_object<multiplies<T>> : std::true_type {};
template <typename T>
struct has_known_identity<multiplies<T>, T> : std::bool_constant<is_arithmetic_v<T>> {};
template <Arithmetic T>
struct known_identity<multiplies<T>, T> {
    static constexpr T value = T{1};
};

// boolean logic

template <typename T = void>
struct bit_and {
    T operator()(const T &x, const T &y) const { return x & y; }
};
template <typename T>
struct is_function_object<bit_and<T>> : std::true_type {};
template <typename T>
struct has_known_identity<bit_and<T>, T> : std::bool_constant<std::is_integral_v<T>> {};
template <std::integral T>
struct known_identity<bit_and<T>, T> {
    static constexpr T value = ~T{};
};

template <typename T = void>
struct bit_or {
    T operator()(const T &x, const T &y) const { return x | y; }
};
template <typename T>
struct is_function_object<bit_or<T>> : std::true_type {};
template <typename T>
struct has_known_identity<bit_or<T>, T> : std::bool_constant<std::is_integral_v<T>> {};
template <std::integral T>
struct known_identity<bit_or<T>, T> {
    static constexpr T value = T{};
};

template <typename T = void>
struct bit_xor {
    T operator()(const T &x, const T &y) const { return x ^ y; }
};
template <typename T>
struct is_function_object<bit_xor<T>> : std::true_type {};
template <typename T>
struct has_known_identity<bit_xor<T>, T> : std::bool_constant<std::is_integral_v<T>> {};
template <std::integral T>
struct known_identity<bit_xor<T>, T> {
    static constexpr T value = T{};
};

// logic

template <typename T = void>
struct logical_and {
    T operator()(const T &x, const T &y) const { return x && y; }
};
template <typename T>
struct is_function_object<logical_and<T>> : std::true_type {};
template <typename T>
struct has_known_identity<logical_and<T>, T> : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};
template <Boolean T>
struct known_identity<logical_and<T>, T> {
    static constexpr T value = true;
};

template <typename T = void>
struct logical_or {
    T operator()(const T &x, const T &y) const { return x || y; }
};
template <typename T>
struct is_function_object<logical_or<T>> : std::true_type {};
template <typename T>
struct has_known_identity<logical_or<T>, T> : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};
template <Boolean T>
struct known_identity<logical_or<T>, T> {
    static constexpr T value = false;
};

// min/max

template <typename T = void>
struct minimum {
    T operator()(const T &x, const T &y) const { return x <= y ? x : y; }
};
template <typename T>
struct is_function_object<minimum<T>> : std::true_type {};
template <typename T>
struct has_known_identity<minimum<T>, T> : std::bool_constant<is_arithmetic_v<T>> {};
template <Arithmetic T>
struct known_identity<minimum<T>, T> {
    static constexpr T value
        = std::is_floating_point_v<T> ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
};

template <typename T = void>
struct maximum {
    T operator()(const T &x, const T &y) const { return x >= y ? x : y; }
};
template <typename T>
struct is_function_object<maximum<T>> : std::true_type {};
template <typename T>
struct has_known_identity<maximum<T>, T> : std::bool_constant<is_arithmetic_v<T>> {};
template <Arithmetic T>
struct known_identity<maximum<T>, T> {
    static constexpr T value
        = std::is_floating_point_v<T> ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
};

} // namespace simsycl::sycl
