#pragma once

#include "forward.hh"

namespace simsycl::sycl {

template <class T>
struct is_group : std::false_type {};
template <class T>
inline constexpr bool is_group_v = is_group<T>::value;

template <class T>
struct is_sub_group : std::false_type {};
template <class T>
inline constexpr bool is_sub_group_v = is_sub_group<T>::value;

} // namespace simsycl::sycl
