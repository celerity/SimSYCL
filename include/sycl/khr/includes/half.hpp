#pragma once

#include "version.hpp" // IWYU pragma: keep

#include <simsycl/config.hh>
#if !SIMSYCL_FEATURE_HALF_TYPE
#error "Half type is not supported, but half.hpp is included."
#endif

// if SIMSYCL_FEATURE_HALF_TYPE is set, forward.hh declares sycl::half
#include "../../../simsycl/sycl/forward.hh" // IWYU pragma: keep
