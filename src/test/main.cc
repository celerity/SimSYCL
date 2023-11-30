#include "simsycl/sycl/group_functions.hh"
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
            sycl::group_barrier(it.get_group());
            printf("[B] %zu / %zu - wg: %zu (%zu / %zu)\n", it.get_global_id(0), it.get_global_range(0),
                it.get_group(0), it.get_local_id(0), it.get_local_range(0));
        });
    });

    q.submit([](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>{10, 5}, [](sycl::nd_item<1> it) {
            auto ret = sycl::group_broadcast(it.get_group(), 42 + it.get_global_linear_id(), 1);
            printf("broadcasted value at %zu: %zu\n", it.get_global_linear_id(), ret);
            auto ret2 = sycl::group_broadcast(it.get_group(), 31337 + it.get_global_linear_id(), 4);
            printf("broadcasted value at %zu: %zu\n", it.get_global_linear_id(), ret2);
        });
    });
}
