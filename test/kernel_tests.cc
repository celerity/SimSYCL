#include <simsycl/sycl.hh>
#include <simsycl/system.hh>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>


using namespace simsycl;

class test_simple_kernel_name;

template<int Dims>
class test_templated_kernel_name;

TEST_CASE("application kernels can be enumerated through get_kernel_ids()", "[kernel]") {
    SECTION("for inline kernel names") {
        const std::string name = sycl::get_kernel_id<class test_inline_kernel_name>().get_name();
        CHECK_THAT(name, Catch::Matchers::ContainsSubstring("test_inline_kernel_name"));
        sycl::queue().single_task<class test_inline_kernel_name>([] {});
    }

    SECTION("for simple kernel names") {
        const std::string name = sycl::get_kernel_id<test_simple_kernel_name>().get_name();
        CHECK_THAT(name, Catch::Matchers::ContainsSubstring("test_simple_kernel_name"));
    }

    SECTION("for templated kernel names") {
        const std::string name = sycl::get_kernel_id<test_templated_kernel_name<42>>().get_name();
        CHECK_THAT(name, Catch::Matchers::ContainsSubstring("test_templated_kernel_name<42>"));
    }
}

TEST_CASE("launching kernels for enumeration in other test", "[kernel]") {
    sycl::queue().parallel_for<test_simple_kernel_name>(sycl::range<1>(1), [](sycl::item<1>) {});
    sycl::queue().parallel_for<test_templated_kernel_name<42>>(sycl::range<1>(1), [](sycl::item<1>) {});
}
