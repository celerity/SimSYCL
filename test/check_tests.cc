#include "test_utils.hh"

#include <sycl/sycl.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace simsycl;
using Catch::Matchers::ContainsSubstring;

#if SIMSYCL_CHECK_MODE == SIMSYCL_CHECK_NONE
TEST_CASE("SIMSYCL_CHECK follows the configured setting - NONE", "[check]") { //
    SIMSYCL_CHECK(false);
    CHECK(true);
}
#endif

#if SIMSYCL_CHECK_MODE == SIMSYCL_CHECK_LOG

#include <iostream>
#include <sstream>

TEST_CASE("SIMSYCL_CHECK follows the configured setting - LOG", "[check]") {
    auto stdout_buffer = std::cout.rdbuf();
    std::ostringstream oss;
    std::cout.rdbuf(oss.rdbuf());
    SIMSYCL_CHECK(false && "Bla");
    std::cout.rdbuf(stdout_buffer);
    REQUIRE_THAT(oss.str(), ContainsSubstring("SimSYCL check failed: false && \"Bla\" at "));
}

#endif

#if SIMSYCL_CHECK_MODE == SIMSYCL_CHECK_THROW

TEST_CASE("SIMSYCL_CHECK follows the configured setting - THROW", "[check]") {
    REQUIRE_THROWS_WITH(
        [] { SIMSYCL_CHECK(false && "Bla"); }(), ContainsSubstring("SimSYCL check failed: false && \"Bla\" at "));
}

TEST_CASE("Exceptions are propagated out of work items", "[check][exceptions]") {
    sycl::queue q;
    REQUIRE_THROWS_WITH(q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>{2, 2}, [](sycl::nd_item<1>) { SIMSYCL_CHECK(false && "Bla"); });
    }),
        ContainsSubstring("SimSYCL check failed: false && \"Bla\" at "));
}

SIMSYCL_START_IGNORING_DEPRECATIONS

template<typename T>
extern const sycl::access_mode accessor_mode_v;

template<typename T, int D, sycl::access_mode M, sycl::target A>
constexpr sycl::access_mode accessor_mode_v<sycl::accessor<T, D, M, A>> = M;

template<typename T, int D, sycl::access_mode M>
constexpr sycl::access_mode accessor_mode_v<sycl::host_accessor<T, D, M>> = M;

using host_accessor_types = std::tuple<                                               //
    sycl::host_accessor<int, 1, sycl::access_mode::read>,                             //
    sycl::host_accessor<int, 0, sycl::access_mode::read>,                             //
    sycl::host_accessor<int, 1, sycl::access_mode::read_write>,                       //
    sycl::host_accessor<int, 0, sycl::access_mode::read_write>,                       //
    sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::host_buffer>,       //
    sycl::accessor<int, 0, sycl::access_mode::read, sycl::target::host_buffer>,       //
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::host_buffer>, //
    sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::target::host_buffer>>;

using command_group_accessor_types = std::tuple<                                        //
    sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::global_buffer>,       //
    sycl::accessor<int, 0, sycl::access_mode::read, sycl::target::global_buffer>,       //
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::global_buffer>, //
    sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::target::global_buffer>, //
    sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::constant_buffer>,     //
    sycl::accessor<int, 0, sycl::access_mode::read, sycl::target::constant_buffer>>;

using accessor_type_combinations = test::tuple_cross_product<host_accessor_types, command_group_accessor_types>::type;

TEMPLATE_LIST_TEST_CASE("Overlapping lifetimes between host- and command-group accessors are diagnosed", "[check]",
    accessor_type_combinations) {
    using host_accessor_type = std::tuple_element_t<0, TestType>;
    using command_group_accessor_type = std::tuple_element_t<1, TestType>;

    sycl::buffer<int, 1> buf(100);
    host_accessor_type host_acc(buf);

    const auto submit_overlapping_command_group = [&] {
        sycl::queue().submit([&](sycl::handler &cgh) {
            command_group_accessor_type acc(buf, cgh);
            cgh.single_task([=] { (void)acc; });
        });
    };

    if(accessor_mode_v<host_accessor_type> == sycl::access_mode::read
        && accessor_mode_v<command_group_accessor_type> == sycl::access_mode::read) {
        REQUIRE_NOTHROW(submit_overlapping_command_group());
    } else {
        REQUIRE_THROWS_WITH(
            submit_overlapping_command_group(), ContainsSubstring("overlaps with a live host accessor"));
    }
}

SIMSYCL_STOP_IGNORING_DEPRECATIONS

#endif
