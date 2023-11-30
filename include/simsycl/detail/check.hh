#pragma once

#include <cassert>

#define SIMSYCL_CHECK(...) assert(__VA_ARGS__)

#define SIMSYCL_NOT_IMPLEMENTED SIMSYCL_CHECK(false && "Not implemented");

extern void var_use_dummy(...);

#define SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(...)                                                                       \
    if(false) var_use_dummy(__VA_ARGS__);                                                                              \
    SIMSYCL_CHECK(false && "Not implemented");
