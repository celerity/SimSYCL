#include "test_utils.hh"
#include <simsycl/schedule.hh>
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
    "naive racy fibonacci implementation produces expected results only with round-robin schedule", "[schedule]") {
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

    sycl::free(buf, q);
}

TEST_CASE("sub-group based word packing built for 32-element sub-groups breaks for different SG size", "[simulation]") {
    const auto sub_group_size = GENERATE(values<uint32_t>({32, 48, 64}));
    simsycl::test::configure_device_with([=](simsycl::device_config &dev) { dev.sub_group_sizes = {sub_group_size}; });
    CAPTURE(sub_group_size);

    const size_t num_blocks = 8;

    sycl::queue q;
    auto *in_buf = sycl::malloc_shared<uint32_t>(32 * num_blocks, q);
    for(size_t i = 0; i < 32 * num_blocks; ++i) { in_buf[i] = i % 5 == 0 ? 0 : i; }

    auto *out_buf = sycl::malloc_shared<uint32_t>(33 * num_blocks, q);
    q.parallel_for(sycl::nd_range<1>(32 * num_blocks, 32 * num_blocks), [=](sycl::nd_item<1> item) {
        const auto group = item.get_group();
        const auto sg = item.get_sub_group();

        const uint32_t word = in_buf[item.get_global_linear_id()];
        uint32_t non_zero_mask = static_cast<uint32_t>(word != 0) << (31 - sg.get_local_linear_id());
        non_zero_mask = sycl::reduce_over_group(sg, non_zero_mask, sycl::bit_or<uint32_t>());

        const uint32_t out_words = static_cast<uint32_t>(sg.leader()) + static_cast<uint32_t>(word != 0);
        const auto out_pos = sycl::exclusive_scan_over_group(group, out_words, sycl::plus<uint32_t>());

        // item 0 in SG writes the mask, every item writes their word
        if(sg.leader()) {
            out_buf[out_pos] = non_zero_mask;
            out_buf[out_pos + 1] = word;
        } else {
            out_buf[out_pos] = word;
        }
    });
    std::vector<uint32_t> out(out_buf, out_buf + 33 * num_blocks);
    sycl::free(in_buf, q);
    sycl::free(out_buf, q);

    std::vector<uint32_t> expected(33 * num_blocks);
    uint32_t non_zero_mask = 0;
    size_t next_mask_pos = 0;
    size_t next_word_pos = 1;
    for(size_t i = 0; i < 32 * num_blocks; ++i) {
        const uint32_t word = i % 5 == 0 ? 0 : i;
        non_zero_mask = (non_zero_mask << 1) | static_cast<uint32_t>(word != 0);
        if(word != 0) { expected[next_word_pos++] = word; }
        if(i % 32 == 31) {
            expected[next_mask_pos] = non_zero_mask;
            next_mask_pos = next_word_pos++;
            non_zero_mask = 0;
        }
    }

    expected.resize(next_mask_pos);
    out.resize(next_mask_pos);

    if(sub_group_size == 32) {
        // the algorithm was implemented for SG size 32
        CHECK(out == expected);
    } else {
        // it breaks when ran with a SG size != 32
        CHECK(out != expected);
    }
}
