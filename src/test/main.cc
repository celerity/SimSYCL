#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;

    q.submit([](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{10}, [](sycl::item<1> it) { printf("%zu / %zu\n", it[0], it.get_range()[0]); });
    });

    q.submit([](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>{10, 5}, [](sycl::nd_item<1> it) {
            printf("[A] %zu / %zu - wg: %zu (%zu / %zu)\n", it.get_global_id(0), it.get_global_range(0),
                it.get_group(0), it.get_local_id(0), it.get_local_range(0));
            it.barrier();
            printf("[B] %zu / %zu - wg: %zu (%zu / %zu)\n", it.get_global_id(0), it.get_global_range(0),
                it.get_group(0), it.get_local_id(0), it.get_local_range(0));
        });
    });
}
