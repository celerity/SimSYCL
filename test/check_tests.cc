#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

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
#endif