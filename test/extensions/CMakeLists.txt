cmake_minimum_required(VERSION 3.13)
project(extensions_test LANGUAGES CXX)

find_package(SimSYCL)

add_executable(extensions_test extensions_test.cc)
add_sycl_to_target(TARGET extensions_test SIMSYCL_ALL_WARNINGS)