#include <cstdio>

#include <sycl/sycl.hpp>


const size_t MAT_SIZE = 128;

template<typename T>
void set_identity(sycl::queue queue, sycl::buffer<T, 2> mat, bool reverse) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor dw{mat, cgh, sycl::write_only, sycl::no_init};
        const auto range = mat.get_range();

        cgh.parallel_for<class set_identity_kernel>(range, [=](sycl::item<2> item) {
            if(!reverse) {
                dw[item] = item[0] == item[1];
            } else {
                dw[item] = item[0] == (range[1] - item[1] - 1);
            }
        });
    });
}

template<typename T>
void multiply(sycl::queue queue, sycl::buffer<T, 2> mat_a, sycl::buffer<T, 2> mat_b, sycl::buffer<T, 2> mat_c) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor a{mat_a, cgh, sycl::read_only};
        sycl::accessor b{mat_b, cgh, sycl::read_only};
        sycl::accessor c{mat_c, cgh, sycl::write_only, sycl::no_init};

        // Use local-memory tiling to avoid waiting on global memory too often
        // Note: We assume a local range size of 64 here, this should be supported by most devices.
        const size_t group_size = 8;
        sycl::local_accessor<T, 2> scratch_a{{group_size, group_size}, cgh};
        sycl::local_accessor<T, 2> scratch_b{{group_size, group_size}, cgh};

        cgh.parallel_for<class mat_mul>(
            sycl::nd_range<2>{{MAT_SIZE, MAT_SIZE}, {group_size, group_size}}, [=](sycl::nd_item<2> item) {
                T sum{};
                const auto lid = item.get_local_id();
                for(size_t j = 0; j < MAT_SIZE; j += group_size) {
                    scratch_a[lid] = a[item.get_group(0) * group_size + lid[0]][j + lid[1]];
                    scratch_b[lid] = b[j + lid[0]][item.get_group(1) * group_size + lid[1]];
                    sycl::group_barrier(item.get_group());

                    for(size_t k = 0; k < group_size; ++k) {
                        const auto a_ik = scratch_a[lid[0]][k];
                        const auto b_kj = scratch_b[k][lid[1]];
                        sum += a_ik * b_kj;
                    }
                    sycl::group_barrier(item.get_group());
                }
                c[item.get_global_id()] = sum;
            });
    });
}

// TODO this should really reduce into a buffer<bool> on the device, but we currently do not support reductions
template<typename T>
void verify(sycl::queue &queue, sycl::buffer<T, 2> mat_c) {
    queue
        .submit([&](sycl::handler &cgh) {
            sycl::accessor c{mat_c, cgh, sycl::read_only_host_task};
            cgh.host_task([c, range = mat_c.get_range()] {
                for(size_t i = 0; i < range[0]; ++i) {
                    for(size_t j = 0; j < range[1]; ++j) {
                        const float received = c[i][j];
                        const float expected = i == j;
                        if(expected != received) {
                            fprintf(stderr, "Verification failed for element %zu,%zu: %g (received) != %g (expected)\n",
                                i, j, received, expected);
                            return;
                        }
                    }
                }
                fprintf(stderr, "Verification passed\n");
            });
        })
        .wait();
}

int main() {
    sycl::queue queue;

    const auto range = sycl::range<2>(MAT_SIZE, MAT_SIZE);
    sycl::buffer<float, 2> mat_a_buf(range);
    sycl::buffer<float, 2> mat_b_buf(range);
    sycl::buffer<float, 2> mat_c_buf(range);

    set_identity(queue, mat_a_buf, false);
    set_identity(queue, mat_b_buf, true);

    multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
    multiply(queue, mat_b_buf, mat_c_buf, mat_a_buf);

    verify(queue, mat_a_buf);
}
