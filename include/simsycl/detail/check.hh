#pragma once

#include <cassert>

#define SIMSYCL_CHECK(...) assert(__VA_ARGS__)

#define SIMSYCL_NOT_IMPLEMENTED SIMSYCL_CHECK(false && "Not implemented");

#define SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(...)                                                                       \
    (void)(__VA_ARGS__);                                                                                               \
    SIMSYCL_CHECK(false && "Not implemented");
