#pragma once

#include <concepts>

#include "type_traits.hh"

namespace simsycl::sycl {

// Standard C++

template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template <typename T>
concept Fundamental = std::is_fundamental_v<T>;

template <typename T>
concept Pointer = std::is_pointer_v<T>;

template <typename T>
concept PointerToFundamental = Pointer<T> && Fundamental<std::remove_pointer_t<T>>;

// SYCL

template <typename T>
concept Group = is_group_v<T>;

template <typename T>
concept SubGroup = is_sub_group_v<T>;

// TODO check what the standard actually means when it says "SYCL function object type"
template <typename Fn, typename T>
concept BinaryOperation = std::is_invocable_r_v<T, Fn, T, T>;

} // namespace simsycl::sycl