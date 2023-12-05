#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

using namespace simsycl;

TEST_CASE("Group barriers behave as expected", "[group_op]") {
    enum class point { a, b, c };
    struct record {
        point p;
        size_t global_id;
        size_t global_range;
        size_t group_id;
        size_t local_id;
        size_t local_range;
        auto operator<=>(const record &) const = default;
    };
    const std::vector<record> expected = {
        {point::a, 0, 4, 0, 0, 2},
        {point::a, 1, 4, 0, 1, 2},
        {point::a, 2, 4, 1, 0, 2},
        {point::a, 3, 4, 1, 1, 2},
        {point::b, 0, 4, 0, 0, 2},
        {point::b, 1, 4, 0, 1, 2},
        {point::b, 2, 4, 1, 0, 2},
        {point::b, 3, 4, 1, 1, 2},
        {point::c, 0, 4, 0, 0, 2},
        {point::c, 1, 4, 0, 1, 2},
        {point::c, 2, 4, 1, 0, 2},
        {point::c, 3, 4, 1, 1, 2},
    };
    std::vector<record> actual;

    SECTION("For work groups") {
        sycl::queue().submit([&actual](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{4, 2}, [&actual](sycl::nd_item<1> it) {
                actual.emplace_back(point::a, it.get_global_id(0), it.get_global_range(0), it.get_group(0),
                    it.get_local_id(0), it.get_local_range(0));
                sycl::group_barrier(it.get_group());
                actual.emplace_back(point::b, it.get_global_id(0), it.get_global_range(0), it.get_group(0),
                    it.get_local_id(0), it.get_local_range(0));
                sycl::group_barrier(it.get_group());
                actual.emplace_back(point::c, it.get_global_id(0), it.get_global_range(0), it.get_group(0),
                    it.get_local_id(0), it.get_local_range(0));
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 2u};
        sycl::queue().submit([&actual](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{4, 4}, [&actual](sycl::nd_item<1> it) {
                const auto &sg = it.get_sub_group();
                actual.emplace_back(point::a, it.get_global_id(0), it.get_global_range(0), sg.get_group_linear_id(),
                    sg.get_local_linear_id(), sg.get_local_range().size());
                sycl::group_barrier(sg);
                actual.emplace_back(point::b, it.get_global_id(0), it.get_global_range(0), sg.get_group_linear_id(),
                    sg.get_local_linear_id(), sg.get_local_range().size());
                sycl::group_barrier(sg);
                actual.emplace_back(point::c, it.get_global_id(0), it.get_global_range(0), sg.get_group_linear_id(),
                    sg.get_local_linear_id(), sg.get_local_range().size());
            });
        });
    }

    CHECK(actual == expected);
}

TEST_CASE("Group broadcasts behave as expected", "[group_op][broadcast]") {
    std::vector<int> expected = {42, 42, 42, 42, 46, 46, 46, 46};
    std::vector<int> actual(expected.size());

    sycl::queue().submit([&actual](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&actual](sycl::nd_item<1> it) {
            actual[it.get_global_linear_id()]
                = sycl::group_broadcast(it.get_group(), 40 + it.get_global_linear_id(), 2);
        });
    });

    CHECK(actual == expected);
}
