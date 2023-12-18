#pragma once

#include <concepts> // IWYU pragma: keep

#include "type_traits.hh"

namespace simsycl::sycl {

// Standard C++

template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

template<typename T>
concept Fundamental = std::is_fundamental_v<T>;

template<typename T>
concept Pointer = std::is_pointer_v<T>;

template<typename T>
concept PointerToFundamental = Pointer<T> && Fundamental<std::remove_pointer_t<T>>;

template<typename T>
concept Boolean = std::is_same_v<std::remove_cv_t<T>, bool>;

// SYCL

template<typename T>
concept Group = sycl::is_group_v<T>;

template<typename T>
concept SubGroup = detail::is_sub_group_v<T>;

template<typename Fn, typename T>
concept BinaryOperation = std::is_invocable_r_v<T, Fn, T, T>;

template<typename Fn>
concept SyclFunctionObject = detail::is_function_object_v<Fn>;

template<typename T>
concept Arithmetic = detail::is_arithmetic_v<T>;

} // namespace simsycl::sycl
