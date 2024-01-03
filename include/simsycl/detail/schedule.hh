#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "../sycl/device.hh"
#include "../sycl/forward.hh"
#include "../sycl/id.hh"
#include "../sycl/item.hh"
#include "../sycl/kernel.hh"
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


template<typename WorkgroupFunctionType>
void sequential_for_work_group(sycl::range<1> num_work_groups, std::optional<sycl::range<1>> work_group_size,
    const WorkgroupFunctionType &kernel_func) {
    sycl::id<1> group_id;
    for(group_id[0] = 0; group_id[0] < num_work_groups[0]; ++group_id[0]) {
        concurrent_group impl;
        sycl::group<1> group = make_hierarchical_group(make_item(group_id, num_work_groups), work_group_size, &impl);
        kernel_func(group);
    }
}

template<typename WorkgroupFunctionType>
void sequential_for_work_group(sycl::range<2> num_work_groups, std::optional<sycl::range<2>> work_group_size,
    const WorkgroupFunctionType &kernel_func) {
    sycl::id<2> group_id;
    for(group_id[0] = 0; group_id[0] < num_work_groups[0]; ++group_id[0]) {
        for(group_id[1] = 0; group_id[1] < num_work_groups[1]; ++group_id[1]) {
            concurrent_group impl;
            sycl::group<2> group
                = make_hierarchical_group(make_item(group_id, num_work_groups), work_group_size, &impl);
            kernel_func(group);
        }
    }
}

template<typename WorkgroupFunctionType>
void sequential_for_work_group(sycl::range<3> num_work_groups, std::optional<sycl::range<3>> work_group_size,
    const WorkgroupFunctionType &kernel_func) {
    sycl::id<3> group_id;
    for(group_id[0] = 0; group_id[0] < num_work_groups[0]; ++group_id[0]) {
        for(group_id[1] = 0; group_id[1] < num_work_groups[1]; ++group_id[1]) {
            for(group_id[2] = 0; group_id[2] < num_work_groups[2]; ++group_id[2]) {
                concurrent_group impl;
                sycl::group<3> group
                    = make_hierarchical_group(make_item(group_id, num_work_groups), work_group_size, &impl);
                kernel_func(group);
            }
        }
    }
}


struct local_memory_requirement {
    std::unique_ptr<void *> ptr;
    size_t size = 0;
    size_t align = 1;
};


template<int Dimensions>
using nd_kernel = std::function<void(const sycl::nd_item<Dimensions> &)>;

template<int Dimensions>
void cooperative_for_nd_range(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<Dimensions> &kernel);

template<typename KernelName, int Dimensions, typename Offset, typename KernelFunc, typename... Params>
void execute_parallel_for(
    const sycl::range<Dimensions> &range, const Offset &offset, KernelFunc &&func, Params &&...args) {
    register_kernel_on_static_construction<KernelName, KernelFunc>();
    sequential_for(range, offset, func, std::forward<Params>(args)...);
}

template<typename KernelName, int Dimensions, typename KernelFunc, typename... Params>
void execute_parallel_for(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, KernelFunc &&func, Params &&...args) {
    const nd_kernel<Dimensions> kernel(
        [&](const sycl::nd_item<Dimensions> &item) { func(item, std::forward<Params>(args)...); });
    register_kernel_on_static_construction<KernelName, KernelFunc>();
    cooperative_for_nd_range(device, range, local_memory, kernel);
}

template<typename KernelName, typename KernelFunc>
void execute_single_task(KernelFunc &&func) {
    register_kernel_on_static_construction<KernelName, KernelFunc>();
    func();
}

template<typename KernelName, int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_parallel_for(const sycl::range<Dimensions> &range, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    auto &kernel_func = std::get<KernelIndex>(params);
    execute_parallel_for<KernelName>(range, no_offset, kernel_func, std::get<ReductionIndices>(params)...);
}

template<typename KernelName, int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_parallel_for(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const auto &kernel_func = std::get<KernelIndex>(params);
    execute_parallel_for<KernelName>(device, range, local_memory, kernel_func, std::get<ReductionIndices>(params)...);
}

template<typename KernelName, int Dimensions, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(sycl::range<Dimensions> num_work_items, Rest &&...rest) {
    dispatch_parallel_for<KernelName>(num_work_items, std::forward_as_tuple(std::forward<Rest>(rest)...),
        std::make_index_sequence<sizeof...(Rest) - 1>(), std::index_sequence<sizeof...(Rest) - 1>());
}

template<typename KernelName, typename KernelFunc, int Dimensions>
void parallel_for(
    sycl::range<Dimensions> num_work_items, sycl::id<Dimensions> work_item_offset, KernelFunc &&kernel_func) {
    execute_parallel_for<KernelName>(num_work_items, work_item_offset, kernel_func);
}

template<typename KernelName = unnamed_kernel, int Dimensions, typename... Rest,
    std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(const sycl::device &device, sycl::nd_range<Dimensions> execution_range,
    const std::vector<local_memory_requirement> &local_memory, Rest &&...rest) {
    detail::dispatch_parallel_for<KernelName>(device, execution_range, local_memory,
        std::forward_as_tuple(std::forward<Rest>(rest)...), std::make_index_sequence<sizeof...(Rest) - 1>(),
        std::index_sequence<sizeof...(Rest) - 1>());
}

template<typename KernelName, int Dimensions, typename WorkgroupFunctionType>
void parallel_for_work_group(sycl::range<Dimensions> num_work_groups,
    std::optional<sycl::range<Dimensions>> work_group_size, const WorkgroupFunctionType &kernel_func) {
    register_kernel_on_static_construction<KernelName, WorkgroupFunctionType>();
    sequential_for_work_group(num_work_groups, work_group_size, kernel_func);
}

} // namespace simsycl::detail
