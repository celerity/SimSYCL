#pragma once

#include "type_traits.hh"

namespace simsycl::sycl {

template <typename T = void>
struct plus {
    T operator()(const T &x, const T &y) const { return x + y; }
};

template <typename T = void>
struct multiplies {
    T operator()(const T &x, const T &y) const { return x * y; }
};

template <typename T = void>
struct bit_and {
    T operator()(const T &x, const T &y) const { return x & y; }
};

template <typename T = void>
struct bit_or {
    T operator()(const T &x, const T &y) const { return x | y; }
};

template <typename T = void>
struct bit_xor {
    T operator()(const T &x, const T &y) const { return x ^ y; }
};

template <typename T = void>
struct logical_and {
    T operator()(const T &x, const T &y) const { return x && y; }
};

template <typename T = void>
struct logical_or {
    T operator()(const T &x, const T &y) const { return x || y; }
};

template <typename T = void>
struct minimum {
    T operator()(const T &x, const T &y) const { return x <= y ? x : y; }
};

template <typename T = void>
struct maximum {
    T operator()(const T &x, const T &y) const { return x >= y ? x : y; }
};

template <class T>
struct is_function_object<plus<T>> : std::true_type {};

template <class T>
struct is_function_object<multiplies<T>> : std::true_type {};

template <class T>
struct is_function_object<bit_and<T>> : std::true_type {};

template <class T>
struct is_function_object<bit_or<T>> : std::true_type {};

template <class T>
struct is_function_object<bit_xor<T>> : std::true_type {};

template <class T>
struct is_function_object<logical_and<T>> : std::true_type {};

template <class T>
struct is_function_object<logical_or<T>> : std::true_type {};

template <class T>
struct is_function_object<minimum<T>> : std::true_type {};

template <class T>
struct is_function_object<maximum<T>> : std::true_type {};

} // namespace simsycl::sycl
