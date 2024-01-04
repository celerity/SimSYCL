#pragma once

#include "allocation.hh"

#include "../schedule.hh"
#include "../sycl/device.hh"
#include "../sycl/forward.hh"
#include "../sycl/id.hh"
#include "../sycl/item.hh"
#include "../sycl/kernel.hh"
#include "../sycl/nd_item.hh"
#include "../sycl/nd_range.hh"
#include "../sycl/range.hh"

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>


namespace simsycl::detail {

struct no_offset_t {
} inline constexpr no_offset;

template<typename>
struct with_offset;
template<>

struct with_offset<no_offset_t> : std::false_type {};
template<int Dimensions>

struct with_offset<sycl::id<Dimensions>> : std::true_type {};

template<typename T>
inline constexpr bool with_offset_v = with_offset<T>::value;


struct local_memory_requirement {
    std::unique_ptr<void *> ptr;
    size_t size = 0;
    size_t align = 1;
};


template<int Dimensions>
using nd_kernel = std::function<void(const sycl::nd_item<Dimensions> &)>;

template<int Dimensions, bool WithOffset>
using simple_kernel = std::function<void(const sycl::item<Dimensions, WithOffset> &)>;

template<int Dimensions>
using hierarchical_kernel = std::function<void(const sycl::group<Dimensions> &)>;

template<int Dimensions, typename Offset>
void sequential_for(const sycl::range<Dimensions> &range, const Offset &offset,
    const simple_kernel<Dimensions, with_offset_v<Offset>> &kernel);

template<int Dimensions>
void sequential_for_work_group(sycl::range<Dimensions> num_work_groups,
    std::optional<sycl::range<Dimensions>> work_group_size, const hierarchical_kernel<Dimensions> &kernel);

template<int Dimensions>
void cooperative_for_nd_range(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<Dimensions> &kernel);

template<typename KernelName, int Dimensions, typename Offset, typename KernelFunc, typename... Params>
void execute_parallel_for(
    const sycl::range<Dimensions> &range, const Offset &offset, KernelFunc &&func, Params &&...args) {
    register_kernel_on_static_construction<KernelName, KernelFunc>();
    const simple_kernel<Dimensions, with_offset_v<Offset>> kernel(
        [&](const sycl::item<Dimensions> &item) { func(item, std::forward<Params>(args)...); });
    sequential_for(range, offset, kernel);
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

template<int Dimensions>
[[nodiscard]] std::vector<allocation> prepare_hierarchical_parallel_for(const sycl::device &device,
    std::optional<sycl::range<Dimensions>> work_group_size, const std::vector<local_memory_requirement> &local_memory);

template<typename KernelName, int Dimensions, typename WorkgroupFunctionType>
void parallel_for_work_group(const sycl::device &device, sycl::range<Dimensions> num_work_groups,
    std::optional<sycl::range<Dimensions>> work_group_size, const std::vector<local_memory_requirement> &local_memory,
    const WorkgroupFunctionType &kernel_func) {
    register_kernel_on_static_construction<KernelName, WorkgroupFunctionType>();
    const auto local_allocations = prepare_hierarchical_parallel_for(device, work_group_size, local_memory);
    sequential_for_work_group(num_work_groups, work_group_size, hierarchical_kernel<Dimensions>(kernel_func));
}

} // namespace simsycl::detail
