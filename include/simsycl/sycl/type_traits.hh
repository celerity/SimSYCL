#pragma once

#include <bit>

namespace simsycl::sycl {

template<class T>
struct is_group : std::false_type {};
template<class T>
inline constexpr bool is_group_v = is_group<T>::value;

template<typename BinaryOperation, typename AccumulatorT>
struct known_identity {};
template<typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v = known_identity<BinaryOperation, AccumulatorT>::value;

template<typename BinaryOperation, typename AccumulatorT>
struct has_known_identity : std::false_type {};
template<typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v = has_known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace simsycl::sycl

namespace simsycl::detail {

template<class T>
struct is_sub_group : std::false_type {};
template<class T>
inline constexpr bool is_sub_group_v = is_sub_group<T>::value;

template<class Fn>
struct is_function_object : std::false_type {};
template<class Fn>
inline constexpr bool is_function_object_v = is_function_object<Fn>::value;

#if SIMSYCL_FEATURE_HALF_TYPE
template<typename T>
struct is_arithmetic : std::bool_constant<std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>> {};
#else
using std::is_arithmetic;
#endif

template<class T>
inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

template<typename T>
struct is_floating_point
#if SIMSYCL_FEATURE_HALF_TYPE
    : std::bool_constant<std::is_same_v<T, sycl::half> || std::is_same_v<T, float> || std::is_same_v<T, double>> {
#else
    : std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, double>> {
#endif
};

template<class T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template<typename...>
constexpr bool always_false = false;

} // namespace simsycl::detail

namespace simsycl::sycl {

// TODO consider moving this to a different header.
using std::bit_cast;

// approximation. must inherit to allow specialization
template<typename T>
struct is_device_copyable : std::is_nothrow_copy_constructible<T> {};

template<typename T>
inline constexpr bool is_device_copyable_v = is_device_copyable<T>::value;

} // namespace simsycl::sycl