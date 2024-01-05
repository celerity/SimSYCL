Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests
  test_utils.hh
  test_setup.cc
  ambiguity_tests.cc
  group_op_tests.cc
  hierarchical_tests.cc
  kernel_tests.cc
  launch_tests.cc
  marray_tests.cc
  math_tests.cc
  reduction_tests.cc
  simulation_tests.cc
  vec_tests.cc
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain simsycl)
set_simsycl_target_options(tests)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)

# we need to check the different compile-time check modes

function(add_check_test check_name check_mode)
  add_executable(test_check_${check_name} check_tests.cc)
  set_simsycl_target_options(test_check_${check_name})
  target_link_libraries(test_check_${check_name} PRIVATE Catch2::Catch2WithMain simsycl)
  target_compile_definitions(test_check_${check_name} PRIVATE SIMSYCL_CHECK_MODE=${check_mode})
  catch_discover_tests(test_check_${check_name})
endfunction()

add_check_test(none SIMSYCL_CHECK_NONE)
add_check_test(log SIMSYCL_CHECK_LOG)
add_check_test(throw SIMSYCL_CHECK_THROW)
