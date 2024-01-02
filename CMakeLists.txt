cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0135 NEW) # ExternalProject downloads will have archive timestamps

set(SIMSYCL_VERSION 0.1)
set(SIMSYCL_VERSION_MAJOR 0)
set(SIMSYCL_VERSION_MINOR 1)
set(SIMSYCL_VERSION_PATCH 0)

project(SimSYCL VERSION "${SIMSYCL_VERSION}" LANGUAGES CXX)

message(STATUS "   _____ _         ______   ______ _     ")
message(STATUS "  / ___/(_)___ ___/ ___\\ \\ / / ___| |    ")
message(STATUS "  \\__ \\/ / __ `__ \\___ \\\\ V / |   | |    ")
message(STATUS " ___/ / / / / / / /___) || || |___| |___ ")
message(STATUS "/____/_/_/ /_/ /_/_____/ |_| \\____|_____|  v${SIMSYCL_VERSION}")
message(STATUS "")
message(STATUS "\"Technically correct is the best kind of correct\"")
message(STATUS "")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_GENERATOR STREQUAL "Ninja")
  # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
  # Rationale in https://github.com/ninja-build/ninja/issues/814
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

find_package(Boost 1.70 COMPONENTS context REQUIRED)

include(FetchContent)

set(LIBENVPP_INSTALL ON CACHE BOOL "" FORCE) # If installation is desired.
FetchContent_Declare(libenvpp
    GIT_REPOSITORY https://github.com/ph3at/libenvpp.git
    GIT_TAG v1.4.0
)
FetchContent_MakeAvailable(libenvpp)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install ON CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

include(CheckTypeSize)
check_type_size(_Float16 FLOAT16 BUILTIN_TYPES_ONLY LANGUAGE CXX)
if (HAVE_FLOAT16)
  set(SIMSYCL_FEATURE_HALF_TYPE 1)
else ()
  set(SIMSYCL_FEATURE_HALF_TYPE 0)
endif()

set(SIMSYCL_ENABLE_ASAN OFF CACHE BOOL "Whether to enable address sanitizer")
set(SIMSYCL_CHECK_MODE "SIMSYCL_CHECK_ABORT" CACHE STRING "Check mode to use")

set(CONFIG_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/simsycl/config.hh")
configure_file(
  "${PROJECT_SOURCE_DIR}/include/simsycl/config.hh.in"
  ${CONFIG_PATH}
  @ONLY
)

# Function to set properties, compile options, and link options for all simsycl targets
function(set_simsycl_target_options target)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20)
    set_target_properties(${target} PROPERTIES CXX_STANDARD_REQUIRED ON)
    target_compile_options(${target} PRIVATE
        # 4180 is a false positive in MSVC in group_operation_impl -- "fixing" it actually breaks the code
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /wd4180>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic $<$<BOOL:${SIMSYCL_ENABLE_ASAN}>:-fsanitize=address>>
    )
    target_link_options(${target} PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:$<$<BOOL:${SIMSYCL_ENABLE_ASAN}>:-fsanitize=address>>
    )
endfunction()

add_library(simsycl
    include/sycl/sycl.hpp
    include/CL/sycl.hpp
    include/simsycl/sycl.hh
    include/simsycl/detail/allocation.hh
    include/simsycl/detail/check.hh
    include/simsycl/detail/coordinate.hh
    include/simsycl/detail/hash.hh
    include/simsycl/detail/schedule.hh
    include/simsycl/detail/subscript.hh
    include/simsycl/detail/utils.hh
    include/simsycl/detail/vec_swizzles.inc
    include/simsycl/sycl/accessor.hh
    include/simsycl/sycl/allocator.hh
    include/simsycl/sycl/async_handler.hh
    include/simsycl/sycl/atomic_fence.hh
    include/simsycl/sycl/atomic_ref.hh
    include/simsycl/sycl/backend.hh
    include/simsycl/sycl/buffer.hh
    include/simsycl/sycl/concepts.hh
    include/simsycl/sycl/context.hh
    include/simsycl/sycl/device.hh
    include/simsycl/sycl/device_selector.hh
    include/simsycl/sycl/enums.hh
    include/simsycl/sycl/event.hh
    include/simsycl/sycl/exception.hh
    include/simsycl/sycl/forward.hh
    include/simsycl/sycl/group_algorithms.hh
    include/simsycl/sycl/group_functions.hh
    include/simsycl/sycl/group.hh
    include/simsycl/sycl/h_item.hh
    include/simsycl/sycl/handler.hh
    include/simsycl/sycl/id.hh
    include/simsycl/sycl/info.hh
    include/simsycl/sycl/interop_handle.hh
    include/simsycl/sycl/image.hh
    include/simsycl/sycl/item.hh
    include/simsycl/sycl/kernel.hh
    include/simsycl/sycl/math.hh
    include/simsycl/sycl/math_geometric.hh
    include/simsycl/sycl/marray.hh
    include/simsycl/sycl/multi_ptr.hh
    include/simsycl/sycl/nd_item.hh
    include/simsycl/sycl/nd_range.hh
    include/simsycl/sycl/binary_ops.hh
    include/simsycl/sycl/platform.hh
    include/simsycl/sycl/private_memory.hh
    include/simsycl/sycl/property.hh
    include/simsycl/sycl/queue.hh
    include/simsycl/sycl/range.hh
    include/simsycl/sycl/reduction.hh
    include/simsycl/sycl/sub_group.hh
    include/simsycl/sycl/type_traits.hh
    include/simsycl/sycl/usm.hh
    include/simsycl/sycl/vec.hh
    include/simsycl/macros.hh
    include/simsycl/system.hh
    ${CONFIG_PATH}
    src/simsycl/check.cc
    src/simsycl/context.cc
    src/simsycl/device.cc
    src/simsycl/schedule.cc
    src/simsycl/platform.cc
    src/simsycl/queue.cc
    src/simsycl/system.cc
    src/simsycl/system_config.cc
)
target_link_libraries(simsycl PRIVATE
    Boost::context
    nlohmann_json::nlohmann_json
    libenvpp::libenvpp
)
target_include_directories(simsycl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Compile with -FPIC to make sure our `inline constexpr` variables work when used in a shared library
set_target_properties(simsycl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_simsycl_target_options(simsycl)

add_executable(main src/test/main.cc)
target_link_libraries(main simsycl)
set_simsycl_target_options(main)

enable_testing()
add_subdirectory(test)
add_subdirectory(examples)

include(CMakePackageConfigHelpers)

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
)
install(
  TARGETS simsycl
  EXPORT install_exports
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/simsycl-config-version.cmake"
  VERSION "${SIMSYCL_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(
  EXPORT install_exports
  FILE "simsycl-targets.cmake"
  NAMESPACE SimSYCL::
  DESTINATION lib/cmake/SimSYCL
)
set(SIMSYCL_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/simsycl-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/simsycl-config.cmake"
  @ONLY
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/simsycl-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/simsycl-config-version.cmake"
  DESTINATION lib/cmake/SimSYCL
)
