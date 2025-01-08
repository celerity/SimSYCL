cmake_minimum_required(VERSION 3.13)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  cmake_policy(SET CMP0135 NEW) # ExternalProject downloads will have archive timestamps
endif()

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

find_package(Boost 1.70 CONFIG COMPONENTS context REQUIRED)

include(FetchContent)

macro(fetch_content_from_remote DEPNAME)
  set(FETCH_REMOTE "${ARGN}")
  if(${DEPNAME}_FOUND)
    message(STATUS "Using system ${DEPNAME} in ${${DEPNAME}_DIR}")
    set(SIMSYCL_DETAIL_USE_SYSTEM_${DEPNAME} ON)
  else()
    list(JOIN FETCH_REMOTE " " FETCH_REMOTE_S)
    message(STATUS "Fetching ${DEPNAME} from ${FETCH_REMOTE_S}")
    set(SIMSYCL_DETAIL_USE_SYSTEM_${DEPNAME} OFF)
    FetchContent_Declare("${DEPNAME}" "${FETCH_REMOTE}")
    FetchContent_MakeAvailable("${DEPNAME}")
  endif()
endmacro()

set(LIBENVPP_INSTALL ON CACHE BOOL "" FORCE) # If installation is desired.
fetch_content_from_remote(libenvpp
    GIT_REPOSITORY https://github.com/ph3at/libenvpp.git
    GIT_TAG v1.5.1
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install ON CACHE INTERNAL "")
fetch_content_from_remote(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)

include(CheckTypeSize)
check_type_size(_Float16 FLOAT16 BUILTIN_TYPES_ONLY LANGUAGE CXX)
set (SIMSYCL_FEATURE_HALF_TYPE ${HAVE_FLOAT16})

option(SIMSYCL_ANNOTATE_SYCL_DEPRECATIONS "Wether to annotate deprecated SYCL APIs with [[deprecated]]" ON)
option(SIMSYCL_ENABLE_ASAN "Whether to enable address sanitizer" OFF)
set(SIMSYCL_CHECK_MODE "ABORT" CACHE STRING "Runtime assertion handling NONE|LOG|THROW|ABORT")

set(CONFIG_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/simsycl/config.hh")
configure_file(
  "${PROJECT_SOURCE_DIR}/include/simsycl/config.hh.in"
  ${CONFIG_PATH}
  @ONLY
)

set(SIMSYCL_LIBRARY simsycl)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/AddToTarget.cmake")

add_library(simsycl
    include/sycl/sycl.hpp
    include/CL/sycl.hpp
    include/simsycl/sycl.hh
    include/simsycl/detail/allocation.hh
    include/simsycl/detail/check.hh
    include/simsycl/detail/coordinate.hh
    include/simsycl/detail/hash.hh
    include/simsycl/detail/math_utils.hh
    include/simsycl/detail/parallel_for.hh
    include/simsycl/detail/subscript.hh
    include/simsycl/detail/utils.hh
    include/simsycl/detail/vec_swizzles.inc
    include/simsycl/sycl/accessor.hh
    include/simsycl/sycl/allocator.hh
    include/simsycl/sycl/async_handler.hh
    include/simsycl/sycl/atomic_fence.hh
    include/simsycl/sycl/atomic_ref.hh
    include/simsycl/sycl/backend.hh
    include/simsycl/sycl/binary_ops.hh
    include/simsycl/sycl/buffer.hh
    include/simsycl/sycl/concepts.hh
    include/simsycl/sycl/context.hh
    include/simsycl/sycl/device_selector.hh
    include/simsycl/sycl/device.hh
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
    include/simsycl/sycl/image.hh
    include/simsycl/sycl/image_accessor.hh
    include/simsycl/sycl/info.hh
    include/simsycl/sycl/interop_handle.hh
    include/simsycl/sycl/item.hh
    include/simsycl/sycl/kernel.hh
    include/simsycl/sycl/marray.hh
    include/simsycl/sycl/math_common.hh
    include/simsycl/sycl/math_geometric.hh
    include/simsycl/sycl/math.hh
    include/simsycl/sycl/multi_ptr.hh
    include/simsycl/sycl/nd_item.hh
    include/simsycl/sycl/nd_range.hh
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
    include/simsycl/schedule.hh
    include/simsycl/system.hh
    ${CONFIG_PATH}
    src/simsycl/check.cc
    src/simsycl/context.cc
    src/simsycl/device.cc
    src/simsycl/group_operation_impl.cc
    src/simsycl/kernel.cc
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
set_simsycl_target_options(simsycl ALL_WARNINGS)

add_executable(main src/test/main.cc)
add_sycl_to_target(TARGET main SIMSYCL_ALL_WARNINGS)

enable_testing()
add_subdirectory(test)

# examples/CMakeLists.txt must support installed SimSYCL as well, so we disable its find_package from here
set(CMAKE_DISABLE_FIND_PACKAGE_SimSYCL TRUE)
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
  "${PROJECT_SOURCE_DIR}/cmake/AddToTarget.cmake"
  DESTINATION lib/cmake/SimSYCL
)
