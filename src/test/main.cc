#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;
    q.submit([](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{10}, [](sycl::item<1> it) { printf("%zu / %zu\n", it[0], it.get_range()[0]); });
    });
}
