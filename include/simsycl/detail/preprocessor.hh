#pragma once

#include <simsycl/config.hh>

// it is frequently necessary to use deprecated functionality to implment deprecated functionality
// this macro is used to suppress warnings in these cases, in a cross-platform way

#if defined(_MSC_VER)
#define SIMSYCL_START_IGNORING_DEPRECATIONS __pragma(warning(push)) __pragma(warning(disable : 4996))
#define SIMSYCL_STOP_IGNORING_DEPRECATIONS __pragma(warning(pop))
#else
#define SIMSYCL_START_IGNORING_DEPRECATIONS                                                                            \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define SIMSYCL_STOP_IGNORING_DEPRECATIONS _Pragma("GCC diagnostic pop")
#endif

#if SIMSYCL_ANNOTATE_SYCL_DEPRECATIONS
#define SIMSYCL_DETAIL_DEPRECATED_IN_SYCL [[deprecated("deprecated in SYCL 2020")]]
#define SIMSYCL_DETAIL_DEPRECATED_IN_SYCL_V(message) [[deprecated("deprecated in SYCL 2020: " message)]]
#else
#define SIMSYCL_DETAIL_DEPRECATED_IN_SYCL
#define SIMSYCL_DETAIL_DEPRECATED_IN_SYCL_V(message)
#endif
