#pragma once

#include "../sycl/id.hh"
#include "../sycl/range.hh"

#include <cstring>

namespace simsycl::detail {

inline size_t get_linear_index(const sycl::range<1> &range, const sycl::id<1> &index) { return index[0]; }

inline size_t get_linear_index(const sycl::range<2> &range, const sycl::id<2> &index) {
    return index[0] * range[1] + index[1];
}

inline size_t get_linear_index(const sycl::range<3> &range, const sycl::id<3> &index) {
    return index[0] * range[1] * range[2] + index[1] * range[2] + index[2];
}

void memcpy_strided_host(const void *source_base_ptr, void *target_base_ptr, size_t elem_size,
    const sycl::range<1> &source_range, const sycl::id<1> &source_offset, const sycl::range<1> &target_range,
    const sycl::id<1> &target_offset, const sycl::range<1> &copy_range) {
    const size_t line_size = elem_size * copy_range[0];
    ::memcpy(static_cast<std::byte *>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
        static_cast<const std::byte *>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset),
        line_size);
}

void memcpy_strided_host(const void *source_base_ptr, void *target_base_ptr, size_t elem_size,
    const sycl::range<2> &source_range, const sycl::id<2> &source_offset, const sycl::range<2> &target_range,
    const sycl::id<2> &target_offset, const sycl::range<2> &copy_range) {
    const size_t line_size = elem_size * copy_range[1];
    const auto source_base_offset = get_linear_index(source_range, source_offset);
    const auto target_base_offset = get_linear_index(target_range, target_offset);
    for(size_t i = 0; i < copy_range[0]; ++i) {
        ::memcpy(static_cast<std::byte *>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
            static_cast<const std::byte *>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]),
            line_size);
    }
}

void memcpy_strided_host(const void *source_base_ptr, void *target_base_ptr, size_t elem_size,
    const sycl::range<3> &source_range, const sycl::id<3> &source_offset, const sycl::range<3> &target_range,
    const sycl::id<3> &target_offset, const sycl::range<3> &copy_range) {
    // We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again
    // during the 2D copy.
    const auto source_base_offset = get_linear_index(source_range, source_offset)
        - get_linear_index(sycl::range<2>{source_range[1], source_range[2]}, {source_offset[1], source_offset[2]});
    const auto target_base_offset = get_linear_index(target_range, target_offset)
        - get_linear_index(sycl::range<2>{target_range[1], target_range[2]}, {target_offset[1], target_offset[2]});
    for(size_t i = 0; i < copy_range[0]; ++i) {
        const auto *const source_ptr = static_cast<const std::byte *>(source_base_ptr)
            + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
        auto *const target_ptr = static_cast<std::byte *>(target_base_ptr)
            + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
        memcpy_strided_host(source_ptr, target_ptr, elem_size, sycl::range<2>{source_range[1], source_range[2]},
            {source_offset[1], source_offset[2]}, {target_range[1], target_range[2]},
            {target_offset[1], target_offset[2]}, {copy_range[1], copy_range[2]});
    }
}

} // namespace simsycl::detail
