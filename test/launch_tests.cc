#include "test_utils.hh"

#include <simsycl/sycl.hh>
#include <simsycl/system.hh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>


using namespace simsycl;

TEMPLATE_TEST_CASE_SIG("groups have distinct local memories", "[launch]", ((int Dims), Dims), 1, 2, 3) {
    sycl::range<Dims> global_range;
    sycl::range<Dims> local_range;
    for(int d = 0; d < Dims; ++d) {
        global_range[d] = 4;
        local_range[d] = 2;
    }

    sycl::queue()
        .submit([&](sycl::handler &cgh) {
            sycl::local_accessor<size_t> a{local_range.size(), cgh};
            cgh.parallel_for(sycl::nd_range(global_range, local_range), [=](sycl::nd_item<Dims> it) {
                const auto my_index = it.get_local_linear_id();
                const auto my_value = 1 + it.get_global_linear_id();
                a[my_index] = my_value;
                sycl::group_barrier(it.get_group());

                const auto buddy_index = it.get_local_linear_id() ^ 1;
                const auto buddy_value = 1 + (it.get_global_linear_id() ^ 1);
                CHECK(a[my_index] == my_value);
                CHECK(a[buddy_index] == buddy_value);
            });
        })
        .wait();
}

TEMPLATE_TEST_CASE_SIG(
    "parallel_for constructs nd_items with correct geometry", "[launch]", ((int Dims), Dims), 1, 2, 3) {
    sycl::range<Dims> global_range;
    sycl::range<Dims> local_range;
    for(int d = 0; d < Dims; ++d) {
        global_range[d] = d * (2 + d);
        local_range[d] = 2 + d;
    }

    std::vector<bool> visited(global_range.size(), false);
    sycl::queue()
        .submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range(global_range, local_range), [=, &visited](sycl::nd_item<Dims> it) {
                const auto global_id = it.get_global_id();
                const auto local_id = it.get_local_id();
                const auto group_id = it.get_group().get_group_id();

                const auto global_linear_id = it.get_global_linear_id();
                CHECK(global_linear_id < global_range.size());
                CHECK(global_linear_id == get_linear_index(global_range, global_id));

                const auto local_linear_id = it.get_local_linear_id();
                CHECK(local_linear_id < local_range.size());
                CHECK(local_linear_id == get_linear_index(local_range, local_id));

                const auto group_range = it.get_group_range();
                const auto group_linear_id = it.get_group_linear_id();
                CHECK(group_linear_id < group_range.size());
                CHECK(group_linear_id == get_linear_index(group_range, group_id));

                const auto sub_group_local_id = it.get_sub_group().get_local_id();
                const auto sub_group_group_id = it.get_sub_group().get_group_id();
                const auto sub_group_range = it.get_sub_group().get_group_range();
                CHECK(sycl::id(local_linear_id) == sub_group_group_id * sycl::id(sub_group_range) + sub_group_local_id);

                CHECK(!visited[global_linear_id]);
                visited[global_linear_id] = true;
            });
        })
        .wait();

    for(size_t i = 0; i < global_range.size(); ++i) { CHECK(visited[i]); }
}

TEST_CASE(
    "parallel_for(nd_range) correctly will re-use fibers and local allocations when the number of groups is large",
    "[launch]") {
    // we currently allocate #max_compute_units groups worth of fibers
    test::configure_device_with([](simsycl::device_config &device) { device.max_compute_units = 2; });

    sycl::range<1> global_range(256);
    sycl::range<1> local_range(16);

    std::vector<bool> visited(global_range.size(), false);
    sycl::queue().parallel_for(sycl::nd_range(global_range, local_range), [=, &visited](sycl::nd_item<1> it) {
        CHECK(!visited[it.get_global_id()]);
        visited[it.get_global_id()] = true;
    });

    for(size_t i = 0; i < global_range.size(); ++i) { CHECK(visited[i]); }
}

TEST_CASE("partial sub-groups are generated if local size is not divisible by device sub-group size", "[launch]") {
    const size_t local_size = GENERATE(values<size_t>({3, 10}));
    const size_t num_groups = 3;
    const size_t global_size = num_groups * local_size;
    const size_t max_sub_group_size = 4;

    CAPTURE(local_size);

    // this is the default, but better be safe and forward-compatible
    test::configure_device_with([=](simsycl::device_config &device) { device.sub_group_sizes = {max_sub_group_size}; });

    std::vector<bool> visited(global_size);
    sycl::queue().parallel_for(sycl::nd_range<1>(global_size, local_size), [&](sycl::nd_item<1> it) {
        auto group = it.get_group();
        CHECK(group.get_local_range() == local_size);

        auto sg = it.get_sub_group();
        if(sg.get_group_linear_id() == sg.get_group_linear_range() - 1) {
            CHECK(sg.get_local_range().size() == local_size % max_sub_group_size);
        } else {
            CHECK(sg.get_local_range().size() == max_sub_group_size);
        }
        CHECK(sg.get_local_linear_id() == it.get_local_linear_id() % max_sub_group_size);
        CHECK(sg.get_max_local_range().size() == max_sub_group_size);

        // barriers should continue to work when partial sub-groups are involved
        sycl::group_barrier(sg);
        sycl::group_barrier(group);

        CHECK(it.get_global_linear_id() < global_size);
        CHECK(it.get_global_linear_id() == it.get_group_linear_id() * local_size + it.get_local_linear_id());
        CHECK(!visited[it.get_global_linear_id()]);
        visited[it.get_global_linear_id()] = true;
    });
}
