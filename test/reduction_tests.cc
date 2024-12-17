#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

using namespace simsycl;


struct basic_parallel_for_wrapper {
    static sycl::range<1> make_range(size_t size) { return {size}; }
    static size_t get_linear_id(sycl::item<1> it) { return it.get_linear_id(); }
};

struct nd_range_parallel_for_wrapper {
    static sycl::nd_range<1> make_range(size_t size) { return {size, 1}; }
    static size_t get_linear_id(sycl::nd_item<1> it) { return it.get_global_linear_id(); }
};


TEMPLATE_TEST_CASE(
    "reductions can be passed to kernels", "[reduction]", basic_parallel_for_wrapper, nd_range_parallel_for_wrapper) {
    sycl::buffer<int> plus_buf(1);
    int mult_var = 99;
    sycl::buffer<int, 2> bit_and_buf({1, 1});
    int bit_or_var = 128;
    sycl::buffer<int, 3> bit_xor_buf({1, 1, 1});
    float min_var = 0;
    float max_var = 0;

    sycl::queue().submit([&](sycl::handler &cgh) {
        sycl::accessor plus(plus_buf, cgh, sycl::write_only, sycl::no_init);
        cgh.parallel_for(sycl::range(1), [=](sycl::item<1> it) { plus[it] = 100; });
    });

    sycl::queue()
        .submit([&](sycl::handler &cgh) {
            cgh.parallel_for(TestType::make_range(10), //
                sycl::reduction(plus_buf, cgh, sycl::plus<int>{}),
                sycl::reduction(&mult_var, sycl::multiplies<>{}, sycl::property::reduction::initialize_to_identity{}),
                sycl::reduction(
                    bit_and_buf, cgh, sycl::bit_and<>{}, sycl::property::reduction::initialize_to_identity{}),
                sycl::reduction(&bit_or_var, sycl::bit_or<>{}),
                sycl::reduction(
                    bit_xor_buf, cgh, sycl::bit_xor<>{}, sycl::property::reduction::initialize_to_identity{}),
                sycl::reduction(&min_var, sycl::minimum<>{}, sycl::property::reduction::initialize_to_identity{}),
                sycl::reduction(&max_var, sycl::maximum<>{}, sycl::property::reduction::initialize_to_identity{}),
                [=](auto item, auto &plus, auto &mult, auto &bit_and, auto &bit_or, auto &bit_xor, auto &min,
                    auto &max) {
                    const int linear_id = static_cast<int>(TestType::get_linear_id(item));
                    plus += linear_id;
                    mult *= 1 + linear_id;
                    bit_and &= 16 + linear_id;
                    bit_or |= linear_id;
                    bit_xor ^= linear_id;
                    min.combine(5.0f - static_cast<float>(linear_id));
                    max.combine(static_cast<float>(linear_id));
                });
        })
        .wait();

    CHECK(*detail::get_buffer_state(plus_buf).data == 100 + 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9);
    CHECK(mult_var == 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10);
    CHECK(*detail::get_buffer_state(bit_and_buf).data == 16);
    CHECK(bit_or_var == (128 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9));
    CHECK(*detail::get_buffer_state(bit_xor_buf).data == (0 ^ 1 ^ 2 ^ 3 ^ 4 ^ 5 ^ 6 ^ 7 ^ 8 ^ 9));
    CHECK(min_var == -4.0f);
    CHECK(max_var == 9.0f);
}
