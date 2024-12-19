cmake_minimum_required(VERSION 3.13)
project(matmul LANGUAGES CXX)

find_package(SimSYCL)

add_executable(matmul matmul.cc)
add_sycl_to_target(TARGET matmul SIMSYCL_ALL_WARNINGS)
