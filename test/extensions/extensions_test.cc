// This program simply checks if extensions are available in the current configuration.
// It attempts no validation of the extension semantics, this can be done in normal tests guarded by feature macros.

// The intention of this is to CI the extension options, by failing to compile when they are disabled and compiling when
// they are enabled.

#include <sycl/sycl.hpp>

int main() {
    sycl::queue queue;

    // SIMSYCL_ENABLE_SYCL_KHR_QUEUE_FLUSH
    queue.khr_flush();
}
