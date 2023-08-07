#include "simsycl/sycl/queue.hh"
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q;

    q.submit([](sycl::handler &cgh) { cgh.parallel_for<class kernel>(sycl::range<1>{1000}, [](sycl::item<1> it) {}); });
}
