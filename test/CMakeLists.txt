Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests group_op_tests.cc)

set_target_properties(tests PROPERTIES CXX_STANDARD 20)
set_target_properties(tests PROPERTIES CXX_STANDARD_REQUIRED ON)
target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(tests PRIVATE -fsanitize=address)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain simsycl)
target_link_options(tests PRIVATE -fsanitize=address)
