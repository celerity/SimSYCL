#pragma once

#include "forward.hh"

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

template<typename T>
struct is_arithmetic : std::bool_constant<std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>> {};
template<class T>
inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

template<typename T>
struct is_floating_point
    : std::bool_constant<std::is_same_v<T, sycl::half> || std::is_same_v<T, float> || std::is_same_v<T, double>> {};
template<class T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template<typename...>
constexpr bool always_false = false;

} // namespace simsycl::detail

// TODO consider moving this to a different header.
namespace simsycl::sycl {
using std::bit_cast;
}