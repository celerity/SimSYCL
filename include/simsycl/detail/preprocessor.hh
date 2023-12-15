#pragma once

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
