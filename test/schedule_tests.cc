#include <sycl/sycl.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>


bool is_fibonacci(const int *buf, int n) {
    for(int i = 0; i < n; ++i) {
        if(buf[i] != (i < 2 ? i : buf[i - 1] + buf[i - 2])) return false;
    }
    return true;
}

TEST_CASE(
    "naive racy fibonacci implementation is produces expected results only with round-robin schedule", "[schedule]") {
    const bool chaos_mode = GENERATE(values<int>({false, true}));
    if(chaos_mode) { simsycl::set_cooperative_schedule(std::make_unique<simsycl::shuffle_schedule>()); }
    INFO((chaos_mode ? "shuffle schedule" : "round-robin schedule"));

    sycl::queue q;
    auto *buf = sycl::malloc_shared<int>(100, q);

    SECTION("in simple parallel_for") {
        q.parallel_for(sycl::range<1>(100), [=](sycl::item<1> item) {
            const int i = item.get_id(0);
            buf[i] = i < 2 ? i : buf[i - 1] + buf[i - 2];
        });
        CHECK(is_fibonacci(buf, 100) == !chaos_mode);
    }

    SECTION("in nd_range parallel_for") {
        q.parallel_for(sycl::nd_range<1>(100, 100), [=](sycl::nd_item<1> item) {
            const int i = item.get_global_id(0);
            buf[i] = i < 2 ? i : buf[i - 1] + buf[i - 2];
        });
        CHECK(is_fibonacci(buf, 100) == !chaos_mode);
    }
}
