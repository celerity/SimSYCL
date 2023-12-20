#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

using namespace simsycl;

TEST_CASE("Basic hierarchical parallel for functionality",
    "[hierarchical][parallel_for_work_group][parallel_for_work_item]") {
    using namespace sycl;
    queue q;

    int test_total = 0;
    constexpr int value = 7;

    q.submit([&](handler &cgh) {
        // Issue 8 work-groups of 8 work-items each
        cgh.parallel_for_work_group(range<3>(2, 2, 2), range<3>(2, 2, 2), [=, &test_total](group<3> g) {
            // this variable will be instantiated for each work-item separately
            private_memory<int, 3> private_mem(g);

            // Issue parallel work-items.  The number issued per work-group is
            // determined by the work-group size range of parallel_for_work_group.
            // In this case, 8 work-items will execute the parallel_for_work_item
            // body for each of the 8 work-groups, resulting in 64 executions
            // globally/total.
            g.parallel_for_work_item([&](h_item<3> itm) { private_mem(itm) = value; });

            // Carry private value across loops
            g.parallel_for_work_item([&](h_item<3> itm) { test_total += private_mem(itm); });
        });
    });

    CHECK(test_total == 8 * 8 * value);
}
