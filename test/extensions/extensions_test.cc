// This program simply checks if extensions are available in the current configuration.
// It attempts no validation of the extension semantics, this can be done in normal tests guarded by feature macros.

// The intention of this is to CI the extension options, by failing to compile when they are disabled and compiling when
// they are enabled.

#include <sycl/sycl.hpp>

int main() {
    sycl::queue queue;

    // SIMSYCL_ENABLE_SYCL_KHR_QUEUE_FLUSH
    queue.khr_flush();

    // SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(1024, 64), [=](sycl::nd_item<1>) {
            [[maybe_unused]] const auto item = sycl::khr::this_nd_item<1>();
            [[maybe_unused]] const auto group = sycl::khr::this_group<1>();
            [[maybe_unused]] const auto sub_group = sycl::khr::this_sub_group();
        });
    });
}
