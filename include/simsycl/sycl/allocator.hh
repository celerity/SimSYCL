#pragma once

#include <cstdlib>
#include <memory>

namespace simsycl::detail {} // namespace simsycl::detail

namespace simsycl::sycl {

template<class T>
using buffer_allocator = std::allocator<T>;

using image_allocator = std::allocator<std::byte>;

} // namespace simsycl::sycl
