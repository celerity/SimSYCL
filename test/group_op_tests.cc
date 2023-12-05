#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

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

    CHECK(actual == expected);
}
