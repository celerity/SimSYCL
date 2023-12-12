#pragma once

#include "forward.hh"

namespace simsycl::sycl {

template <class T>
struct is_group : std::false_type {};
template <class T>
inline constexpr bool is_group_v = is_group<T>::value;

}

namespace simsycl::detail {

template <class T>
struct is_sub_group : std::false_type {};
template <class T>
inline constexpr bool is_sub_group_v = is_sub_group<T>::value;

template <class Fn>
struct is_function_object : std::false_type {};
template <class Fn>
inline constexpr bool is_function_object_v = is_function_object<Fn>::value;

// TODO: Add sycl::half when we have that type
template <typename T>
struct is_arithmetic : std::bool_constant<std::is_arithmetic_v<T>> {};
template <class T>
inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;


} // namespace simsycl::sycl
