cmake_minimum_required(VERSION 3.13)
project(SimSYCL VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost 1.60 COMPONENTS context REQUIRED)

add_library(simsycl
    include/sycl/sycl.hpp
    include/CL/sycl.hpp
    include/simsycl/sycl.hh
    include/simsycl/detail/check.hh
    include/simsycl/detail/config.hh
    include/simsycl/detail/coordinate.hh
    include/simsycl/detail/hash.hh
    include/simsycl/detail/subscript.hh
    include/simsycl/detail/utils.hh
    include/simsycl/sycl/accessor.hh
    include/simsycl/sycl/allocator.hh
    include/simsycl/sycl/backend.hh
    include/simsycl/sycl/buffer.hh
    include/simsycl/sycl/concepts.hh
    include/simsycl/sycl/context.hh
    include/simsycl/sycl/device.hh
    include/simsycl/sycl/enums.hh
    include/simsycl/sycl/event.hh
    include/simsycl/sycl/exception.hh
    include/simsycl/sycl/forward.hh
    include/simsycl/sycl/group_algorithms.hh
    include/simsycl/sycl/group_functions.hh
    include/simsycl/sycl/group.hh
    include/simsycl/sycl/handler.hh
    include/simsycl/sycl/id.hh
    include/simsycl/sycl/image.hh
    include/simsycl/sycl/item.hh
    include/simsycl/sycl/multi_ptr.hh
    include/simsycl/sycl/nd_item.hh
    include/simsycl/sycl/nd_range.hh
    include/simsycl/sycl/platform.hh
    include/simsycl/sycl/property.hh
    include/simsycl/sycl/queue.hh
    include/simsycl/sycl/range.hh
    include/simsycl/sycl/sub_group.hh
    include/simsycl/sycl/type_traits.hh
    src/simsycl/dummy.cc
)
target_link_libraries(simsycl Boost::context)

target_include_directories(simsycl PUBLIC include)
set_target_properties(simsycl PROPERTIES CXX_STANDARD 20)
set_target_properties(simsycl PROPERTIES CXX_STANDARD_REQUIRED ON)
target_compile_options(simsycl PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_options(simsycl PRIVATE -fsanitize=address)
target_link_options(simsycl PRIVATE -fsanitize=address)

add_executable(test src/test/main.cc)
target_link_libraries(test simsycl)
set_target_properties(test PROPERTIES CXX_STANDARD 20)
set_target_properties(test PROPERTIES CXX_STANDARD_REQUIRED ON)
target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_options(test PRIVATE -fsanitize=address)
target_link_options(test PRIVATE -fsanitize=address)
