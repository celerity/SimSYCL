#include "simsycl/sycl/queue.hh"
#include <simsycl/sycl.hh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using Catch::Matchers::ContainsSubstring;

using namespace simsycl;

TEMPLATE_TEST_CASE_SIG(
    "work item queries are correct if supported", "[khr][work_item_queries]", ((int Dims), Dims), 1, 2, 3) {
#if SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES

    sycl::range<Dims> global_range;
    sycl::range<Dims> local_range;
    for(int d = 0; d < Dims; ++d) {
        const int s = d + 1;
        global_range[d] = s * (2 + s);
        local_range[d] = 2 + s;
    }

    std::vector<bool> visited(global_range.size(), false);
    sycl::queue()
        .submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range(global_range, local_range), [=, &visited](sycl::nd_item<Dims> it) {
                const auto global_linear_id = it.get_global_linear_id();
                CHECK(global_linear_id < global_range.size());
                CHECK(!visited[global_linear_id]);
                visited[global_linear_id] = true;

                CHECK(sycl::khr::this_nd_item<Dims>() == it);
                CHECK(sycl::khr::this_group<Dims>() == it.get_group());
                CHECK(sycl::khr::this_sub_group() == it.get_sub_group());

                group_barrier(it.get_group());

                // check again after scheduling through group_barrier
                CHECK(sycl::khr::this_nd_item<Dims>() == it);
                CHECK(sycl::khr::this_group<Dims>() == it.get_group());
                CHECK(sycl::khr::this_sub_group() == it.get_sub_group());
            });
        })
        .wait();

    for(size_t i = 0; i < global_range.size(); ++i) {
        CAPTURE(i);
        CHECK(visited[i]);
    }

#else  // SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES
    SKIP("SYCL_KHR_WORK_ITEM_QUERIES not enabled");
#endif // SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES
}

TEST_CASE("work item queries provide useful errors", "[khr][work_item_queries]") {
#if SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES

    // outside of everything
    REQUIRE_THROWS_WITH(sycl::khr::this_nd_item<1>(), ContainsSubstring("state 'this_nd_item' is not available"));
    REQUIRE_THROWS_WITH(sycl::khr::this_group<1>(), ContainsSubstring("state 'this_group' is not available"));
    REQUIRE_THROWS_WITH(sycl::khr::this_sub_group(), ContainsSubstring("state 'this_sub_group' is not available"));

    // in a non-nd parallel for
    sycl::queue{}.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range{1}, [=](sycl::item<1>) {
            const char *test_str
                = "Make sure that the query originated from a kernel launched with a sycl::nd_range argument";
            CHECK_THROWS_WITH(sycl::khr::this_nd_item<1>(), ContainsSubstring(test_str));
            CHECK_THROWS_WITH(sycl::khr::this_group<1>(), ContainsSubstring(test_str));
            CHECK_THROWS_WITH(sycl::khr::this_sub_group(), ContainsSubstring(test_str));
        });
    });

#else  // SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES
    SKIP("SYCL_KHR_WORK_ITEM_QUERIES not enabled");
#endif // SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES
}
