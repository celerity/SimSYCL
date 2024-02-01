#include <simsycl/detail/allocation.hh>
#include <sycl/sycl.hpp>

#include <catch2/catch_test_macros.hpp>


TEST_CASE("usm_alloc allocates memory of any alignment", "[usm]") {
    const size_t largest_sycl_align_bytes = alignof(sycl::long16);
    const size_t size_bytes = 4096;
    CAPTURE(size_bytes);
    sycl::context ctx;
    for(size_t align_bytes = 1; align_bytes <= largest_sycl_align_bytes; align_bytes *= 2) {
        CAPTURE(align_bytes);
        errno = 0;
        const auto p = simsycl::detail::usm_alloc(ctx, sycl::usm::alloc::host, std::nullopt, size_bytes, align_bytes);
        CHECK(p != nullptr);
        CHECK(reinterpret_cast<std::uintptr_t>(p) % align_bytes == 0);
        simsycl::detail::usm_free(p, ctx);
    }
}
