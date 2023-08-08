cmake_minimum_required(VERSION 3.13)
project(SimSYCL VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(simsycl
    include/sycl/sycl.hpp
    include/CL/sycl.hpp
    include/simsycl/sycl.hh
    include/simsycl/detail/coordinate.hh
    include/simsycl/detail/hash.hh
    include/simsycl/detail/subscript.hh
    include/simsycl/detail/check.hh
    include/simsycl/sycl/range.hh
    include/simsycl/sycl/device.hh
    include/simsycl/sycl/id.hh
    include/simsycl/sycl/multi_ptr.hh
    include/simsycl/sycl/allocator.hh
    include/simsycl/sycl/handler.hh
    include/simsycl/sycl/item.hh
    include/simsycl/sycl/queue.hh
    include/simsycl/sycl/property.hh
    include/simsycl/sycl/image.hh
    include/simsycl/sycl/forward.hh
    include/simsycl/sycl/enums.hh
    include/simsycl/sycl/exception.hh
    include/simsycl/sycl/nd_range.hh
    include/simsycl/sycl/backend.hh
    include/simsycl/sycl/platform.hh
    include/simsycl/sycl/event.hh
    include/simsycl/sycl/accessor.hh
    include/simsycl/sycl/context.hh
    include/simsycl/sycl/buffer.hh
    src/simsycl/dummy.cc
)

target_include_directories(simsycl PUBLIC include)
set_target_properties(simsycl PROPERTIES CXX_STANDARD 17)
set_target_properties(simsycl PROPERTIES CXX_STANDARD_REQUIRED ON)
