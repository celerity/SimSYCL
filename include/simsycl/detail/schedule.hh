#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "../sycl/device.hh"
#include "../sycl/forward.hh"
#include "../sycl/id.hh"
#include "../sycl/item.hh"
#include "../sycl/nd_item.hh"
#include "../sycl/nd_range.hh"
#include "../sycl/range.hh"


namespace simsycl::detail {

struct no_offset_t {
} inline constexpr no_offset;

template<typename Func, typename... Params>
void sequential_for(const sycl::range<1> &range, no_offset_t /* no offset */, Func &&func, Params &&...args) {
    sycl::id<1> id;
    for(id[0] = 0; id[0] < range[0]; ++id[0]) { //
        func(make_item(id, range), std::forward<Params>(args)...);
    }
}

template<typename Func, typename... Params>
void sequential_for(const sycl::range<2> &range, no_offset_t /* no offset */, Func &&func, Params &&...args) {
    sycl::id<2> id;
    for(id[0] = 0; id[0] < range[0]; ++id[0]) {
        for(id[1] = 0; id[1] < range[1]; ++id[1]) { //
            func(make_item(id, range), std::forward<Params>(args)...);
        }
    }
}

template<typename Func, typename... Params>
void sequential_for(const sycl::range<3> &range, no_offset_t /* no offset */, Func &&func, Params &&...args) {
    sycl::id<3> id;
    for(id[0] = 0; id[0] < range[0]; ++id[0]) {
        for(id[1] = 0; id[1] < range[1]; ++id[1]) {
            for(id[2] = 0; id[2] < range[2]; ++id[2]) { //
                func(make_item(id, range), std::forward<Params>(args)...);
            }
        }
    }
}

template<typename Func, typename... Params>
void sequential_for(const sycl::range<1> &range, const sycl::id<1> &offset, Func &&func, Params &&...args) {
    sycl::id<1> id;
    for(id[0] = offset[0]; id[0] < offset[0] + range[0]; ++id[0]) { //
        func(make_item(id, range, offset), std::forward<Params>(args)...);
    }
}

template<typename Func, typename... Params>
void sequential_for(const sycl::range<2> &range, const sycl::id<2> &offset, Func &&func, Params &&...args) {
    sycl::id<2> id;
    for(id[0] = offset[0]; id[0] < offset[0] + range[0]; ++id[0]) {
        for(id[1] = offset[1]; id[1] < offset[1] + range[1]; ++id[1]) { //
            func(make_item(id, range, offset), std::forward<Params>(args)...);
        }
    }
}

template<typename Func, typename... Params>
void sequential_for(const sycl::range<3> &range, const sycl::id<3> &offset, Func &&func, Params &&...args) {
    sycl::id<3> id;
    for(id[0] = offset[0]; id[0] < offset[0] + range[0]; ++id[0]) {
        for(id[1] = offset[1]; id[1] < offset[1] + range[1]; ++id[1]) {
            for(id[2] = offset[2]; id[2] < offset[2] + range[2]; ++id[2]) { //
                func(make_item(id, range, offset), std::forward<Params>(args)...);
            }
        }
    }
}


template<int Dimensions>
sycl::id<Dimensions> linear_index_to_id(const sycl::range<Dimensions> &range, size_t linear_index) {
    sycl::id<Dimensions> id;
    for(int d = Dimensions - 1; d >= 0; --d) {
        id[d] = linear_index % range[d];
        linear_index /= range[d];
    }
    return id;
}


struct local_memory_requirement {
    std::unique_ptr<void *> ptr;
    size_t size = 0;
    size_t align = 1;
};


template<int Dimensions>
using nd_kernel = std::function<void(const sycl::nd_item<Dimensions> &)>;

template<int Dimensions>
void dispatch_for_nd_range(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<Dimensions> &kernel);

template<int Dimensions, typename Func, typename... Params>
    requires(!std::is_same_v<std::remove_cvref_t<Func>, nd_kernel<Dimensions>>)
void dispatch_for_nd_range(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, Func &&func, Params &&...args) {
    const nd_kernel<Dimensions> kernel(
        [&](const sycl::nd_item<Dimensions> &item) { func(item, std::forward<Params>(args)...); });
    dispatch_for_nd_range(device, range, local_memory, kernel);
}

template<int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_for(const sycl::range<Dimensions> &range, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const auto &kernel_func = std::get<KernelIndex>(params);
    detail::sequential_for(range, no_offset, kernel_func, std::get<ReductionIndices>(params)...);
}

template<int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_for(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const auto &kernel_func = std::get<KernelIndex>(params);
    detail::dispatch_for_nd_range(device, range, local_memory, kernel_func, std::get<ReductionIndices>(params)...);
}

template<int Dimensions, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(sycl::range<Dimensions> num_work_items, Rest &&...rest) {
    dispatch_for(num_work_items, std::forward_as_tuple(std::forward<Rest>(rest)...),
        std::make_index_sequence<sizeof...(Rest) - 1>(), std::index_sequence<sizeof...(Rest) - 1>());
}

template<typename KernelType, int Dimensions>
void parallel_for(
    sycl::range<Dimensions> num_work_items, sycl::id<Dimensions> work_item_offset, KernelType &&kernel_func) {
    detail::sequential_for(num_work_items, work_item_offset, kernel_func);
}

template<typename KernelName = unnamed_kernel, int Dimensions, typename... Rest,
    std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(const sycl::device &device, sycl::nd_range<Dimensions> execution_range,
    const std::vector<local_memory_requirement> &local_memory, Rest &&...rest) {
    detail::dispatch_for(device, execution_range, local_memory, std::forward_as_tuple(std::forward<Rest>(rest)...),
        std::make_index_sequence<sizeof...(Rest) - 1>(), std::index_sequence<sizeof...(Rest) - 1>());
}

} // namespace simsycl::detail
