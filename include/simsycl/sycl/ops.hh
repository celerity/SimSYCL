#pragma once

#include "type_traits.hh"

#include <functional>


namespace simsycl::sycl {

using std::bit_and;
using std::bit_or;
using std::bit_xor;
using std::logical_and;
using std::logical_or;
using std::multiplies;
using std::plus;

template <typename T = void>
struct minimum;

template <typename T>
struct minimum {
    T operator()(const T &x, const T &y) const { return x < y ? x : y; }
};

template <>
struct minimum<void> {
    template <typename T, typename U>
    decltype(auto) operator()(T && x, U && y) const {
        return x < y ? std::forward<T>(x) : std::forward<U>(y);
    }
};

template <typename T = void>
struct maximum;

template <typename T>
struct maximum {
    T operator()(const T &x, const T &y) const { return x > y ? x : y; }
};

template <>
struct maximum<void> {
    template <typename T, typename U>
    decltype(auto) operator()(T && x, U && y) const {
        return x > y ? std::forward<T>(x) : std::forward<U>(y);
    }
};

} // namespace simsycl::sycl

namespace simsycl::detail {

template <typename Op, typename AccumulatorT>
struct known_identity {
    static constexpr bool exists = false;
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_arithmetic_v<AccumulatorT> || std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>)
struct known_identity<sycl::plus<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = AccumulatorT{};
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_arithmetic_v<AccumulatorT> || std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>)
struct known_identity<sycl::multiplies<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = AccumulatorT{1};
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_integral_v<AccumulatorT>)
struct known_identity<sycl::bit_and<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = ~AccumulatorT{};
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_integral_v<AccumulatorT>)
struct known_identity<sycl::bit_or<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = AccumulatorT{};
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_integral_v<AccumulatorT>)
struct known_identity<sycl::bit_xor<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = AccumulatorT{};
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>)
struct known_identity<sycl::logical_and<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr bool value = true;
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>)
struct known_identity<sycl::logical_or<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr bool value = false;
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_integral_v<AccumulatorT>)
struct known_identity<sycl::minimum<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = std::numeric_limits<AccumulatorT>::max();
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_floating_point_v<AccumulatorT> || std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>)
struct known_identity<sycl::minimum<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = std::numeric_limits<AccumulatorT>::infinity();
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_integral_v<AccumulatorT>)
struct known_identity<sycl::maximum<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = std::numeric_limits<AccumulatorT>::lowest();
};

template <typename OpT, typename AccumulatorT>
    requires(std::is_floating_point_v<AccumulatorT> || std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>)
struct known_identity<sycl::maximum<OpT>, AccumulatorT> {
    static constexpr bool exists = true;
    static constexpr std::remove_cv_t<AccumulatorT> value = -std::numeric_limits<AccumulatorT>::infinity();
};

template <typename T>
struct is_function_object<sycl::plus<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::multiplies<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::bit_and<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::bit_or<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::bit_xor<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::logical_and<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::logical_or<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::minimum<T>> : std::true_type {};

template <typename T>
struct is_function_object<sycl::maximum<T>> : std::true_type {};


} // namespace simsycl::detail

namespace simsycl::sycl {

template <typename BinaryOperation, typename AccumulatorT>
struct known_identity {
    static constexpr AccumulatorT value = detail::known_identity<BinaryOperation, AccumulatorT>::value;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v = known_identity<BinaryOperation, AccumulatorT>::value;

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity {
    static constexpr bool value = detail::known_identity<BinaryOperation, AccumulatorT>::exists;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v = has_known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace simsycl::sycl
