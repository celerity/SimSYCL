#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

using namespace simsycl;

TEST_CASE("Hierarchical parallel for launches groups", "[hierarchical][parallel_for_work_group]") {
    using namespace sycl;
    queue q;

    constexpr int num_groups = 7;

    SECTION("1D") {
        int test_total = 0;
        std::vector<bool> test_ids(num_groups, false);
        q.submit([&](handler &cgh) {
            cgh.parallel_for_work_group(range<1>(num_groups), [=, &test_total, &test_ids](group<1> g) {
                CHECK(g.get_group_linear_id() < num_groups);
                CHECK(g.get_group_linear_range() == num_groups);
                CHECK(detail::get_group_type(g) == detail::group_type::hierarchical_implicit_size);
                test_total += 1;
                test_ids[g.get_group_linear_id()] = true;
            });
        });
        CHECK(test_total == num_groups);
        CHECK(std::ranges::all_of(test_ids, std::identity{}));
    }

    SECTION("2D") {
        int test_total = 0;
        q.submit([&](handler &cgh) {
            cgh.parallel_for_work_group(range<2>(num_groups, num_groups), [=, &test_total](group<2> g) {
                CHECK(g.get_group_linear_id() < num_groups * num_groups);
                CHECK(g.get_group_linear_range() == num_groups * num_groups);
                CHECK(detail::get_group_type(g) == detail::group_type::hierarchical_implicit_size);
                test_total += 1;
            });
        });
        CHECK(test_total == num_groups * num_groups);
    }

    SECTION("3D") {
        int test_total = 0;
        q.submit([&](handler &cgh) {
            cgh.parallel_for_work_group(range<3>(num_groups, num_groups, num_groups), [=, &test_total](group<3> g) {
                CHECK(g.get_group_linear_id() < num_groups * num_groups * num_groups);
                CHECK(g.get_group_linear_range() == num_groups * num_groups * num_groups);
                CHECK(detail::get_group_type(g) == detail::group_type::hierarchical_implicit_size);
                test_total += 1;
            });
        });
        CHECK(test_total == num_groups * num_groups * num_groups);
    }
}

TEST_CASE("Basic hierarchical parallel for functionality",
    "[hierarchical][parallel_for_work_group][parallel_for_work_item]") {
    using namespace sycl;
    queue q;
    constexpr int value = 7;

    SECTION("1D") {
        int test_total = 0;
        q.submit([&](handler &cgh) {
            cgh.parallel_for_work_group(range<1>(8), range<1>(8), [=, &test_total](group<1> g) {
                private_memory<int, 1> private_mem(g);
                g.parallel_for_work_item([&](h_item<1> itm) { private_mem(itm) = value; });
                g.parallel_for_work_item([&](h_item<1> itm) { test_total += private_mem(itm); });
            });
        });
        CHECK(test_total == 8 * 8 * value);
    }

    SECTION("2D") {
        int test_total = 0;
        q.submit([&](handler &cgh) {
            cgh.parallel_for_work_group(range<2>(4, 2), range<2>(4, 2), [=, &test_total](group<2> g) {
                private_memory<int, 2> private_mem(g);
                g.parallel_for_work_item([&](h_item<2> itm) { private_mem(itm) = value; });
                g.parallel_for_work_item([&](h_item<2> itm) { test_total += private_mem(itm); });
            });
        });
        CHECK(test_total == 4 * 2 * 4 * 2 * value);
    }

    SECTION("3D") {
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
}
