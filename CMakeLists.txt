cmake_minimum_required(VERSION 3.13)
project(SimSYCL VERSION 0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(simsycl
    include/simsycl/sycl/event.hh
    include/simsycl/sycl/handler.hh
    include/simsycl/sycl/queue.hh
    include/simsycl/sycl.hh
    include/CL/sycl.hpp
    include/sycl/sycl.hpp
    src/simsycl/dummy.cc
)

target_include_directories(simsycl PUBLIC include)
set_target_properties(simsycl PROPERTIES CXX_STANDARD_REQUIRED 17)
