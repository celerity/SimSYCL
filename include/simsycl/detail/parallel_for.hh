#pragma once

#include "allocation.hh"

#include "../sycl/device.hh"
#include "../sycl/forward.hh"
#include "../sycl/id.hh"
#include "../sycl/item.hh"
#include "../sycl/kernel.hh"
#include "../sycl/nd_item.hh"
#include "../sycl/nd_range.hh"
#include "../sycl/range.hh"
#include "simsycl/schedule.hh"

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

template<int Dimensions, typename Offset>
auto make_offset_item(const sycl::id<Dimensions> &the_id, const sycl::range<Dimensions> &range, const Offset &offset) {
    if constexpr(std::is_same_v<Offset, no_offset_t>) {
        return make_item(the_id, range);
    } else {
        return make_item(the_id + offset, range, offset);
    }
}

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

template<typename KernelName, int Dimensions, typename Offset, typename KernelFunc, typename... Reducers>
void execute_parallel_for(const sycl::range<Dimensions> &range, const Offset &offset, sycl::kernel_handler kh,
    const KernelFunc &func,
    Reducers &...reducers) //
{
    using item_type = sycl::item<Dimensions, with_offset_v<Offset>>;

    register_kernel_on_static_construction<KernelName, KernelFunc>();

    // directly execute the kernel if the schedule is round robin
    if(dynamic_cast<const round_robin_schedule *>(&get_cooperative_schedule())) {
        if constexpr(std::is_invocable_v<const KernelFunc, item_type, Reducers &..., sycl::kernel_handler>) {
            for_each_id_in_range(range,
                [&](const sycl::id<Dimensions> &id) { func(make_offset_item(id, range, offset), reducers..., kh); });
        } else {
            for_each_id_in_range(
                range, [&](const sycl::id<Dimensions> &id) { func(make_offset_item(id, range, offset), reducers...); });
        }
        return;
    }

    simple_kernel<Dimensions, with_offset_v<Offset>> kernel;
    if constexpr(std::is_invocable_v<const KernelFunc, item_type, Reducers &..., sycl::kernel_handler>) {
        kernel = [&](const item_type &item) { func(item, reducers..., kh); };
    } else {
        static_assert(std::is_invocable_v<const KernelFunc, item_type, Reducers &...>);
        kernel = [&](const item_type &item) { func(item, reducers...); };
    }
    sequential_for(range, offset, kernel);
}

template<typename KernelName, int Dimensions, typename KernelFunc, typename... Reducers>
void execute_parallel_for(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, sycl::kernel_handler kh, const KernelFunc &func,
    Reducers &...reducers) //
{
    register_kernel_on_static_construction<KernelName, KernelFunc>();

    nd_kernel<Dimensions> kernel;
    if constexpr(std::is_invocable_v<const KernelFunc, sycl::nd_item<Dimensions>, Reducers &...,
                     sycl::kernel_handler>) {
        kernel = [&](const sycl::nd_item<Dimensions> &item) { func(item, reducers..., kh); };
    } else {
        static_assert(std::is_invocable_v<const KernelFunc, sycl::nd_item<Dimensions>, Reducers &...>);
        kernel = [&](const sycl::nd_item<Dimensions> &item) { func(item, reducers...); };
    }
    cooperative_for_nd_range(device, range, local_memory, kernel);
}

template<typename KernelName, typename KernelFunc>
void execute_single_task(sycl::kernel_handler kh, KernelFunc &&func) {
    register_kernel_on_static_construction<KernelName, KernelFunc>();
    if constexpr(std::is_invocable_v<const KernelFunc, sycl::kernel_handler>) {
        func(kh);
    } else {
        static_assert(std::is_invocable_v<const KernelFunc>);
        func();
    }
}

template<typename KernelName, int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_parallel_for(const sycl::range<Dimensions> &range, sycl::kernel_handler kh, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    auto &kernel_func = std::get<KernelIndex>(params);
    execute_parallel_for<KernelName>(range, no_offset, kh, kernel_func, std::get<ReductionIndices>(params)...);
}

template<typename KernelName, int Dimensions, typename RestTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_parallel_for(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, sycl::kernel_handler kh, RestTuple &&rest,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const auto &kernel_func = std::get<KernelIndex>(rest);
    execute_parallel_for<KernelName>(device, range, local_memory, kh, kernel_func, std::get<ReductionIndices>(rest)...);
}

template<typename KernelName, int Dimensions, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(sycl::range<Dimensions> num_work_items, sycl::kernel_handler kh, Rest &&...rest) {
    dispatch_parallel_for<KernelName>(num_work_items, kh, std::forward_as_tuple(std::forward<Rest>(rest)...),
        std::make_index_sequence<sizeof...(Rest) - 1>(), std::index_sequence<sizeof...(Rest) - 1>());
}

template<typename KernelName, typename KernelFunc, int Dimensions>
void parallel_for(sycl::range<Dimensions> num_work_items, sycl::id<Dimensions> work_item_offset,
    sycl::kernel_handler kh, const KernelFunc &kernel_func) {
    execute_parallel_for<KernelName>(num_work_items, work_item_offset, kh, kernel_func);
}

template<typename KernelName = unnamed_kernel, int Dimensions, typename... Rest,
    std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(const sycl::device &device, sycl::nd_range<Dimensions> execution_range,
    const std::vector<local_memory_requirement> &local_memory, sycl::kernel_handler kh, Rest &&...rest) {
    detail::dispatch_parallel_for<KernelName>(device, execution_range, local_memory, kh,
        std::forward_as_tuple(std::forward<Rest>(rest)...), std::make_index_sequence<sizeof...(Rest) - 1>(),
        std::index_sequence<sizeof...(Rest) - 1>());
}

template<int Dimensions>
[[nodiscard]] std::vector<allocation> prepare_hierarchical_parallel_for(const sycl::device &device,
    std::optional<sycl::range<Dimensions>> work_group_size, const std::vector<local_memory_requirement> &local_memory);

template<typename KernelName, int Dimensions, typename WorkgroupFunctionType>
void parallel_for_work_group(const sycl::device &device, sycl::range<Dimensions> num_work_groups,
    std::optional<sycl::range<Dimensions>> work_group_size, const std::vector<local_memory_requirement> &local_memory,
    sycl::kernel_handler kh, const WorkgroupFunctionType &kernel_func) //
{
    register_kernel_on_static_construction<KernelName, WorkgroupFunctionType>();

    hierarchical_kernel<Dimensions> kernel;
    if constexpr(std::is_invocable_v<const WorkgroupFunctionType, sycl::group<Dimensions>, sycl::kernel_handler>) {
        kernel = [&](const sycl::group<Dimensions> &group) { kernel_func(group, kh); };
    } else {
        static_assert(std::is_invocable_v<const WorkgroupFunctionType, sycl::group<Dimensions>>);
        kernel = kernel_func;
    }

    const auto local_allocations = prepare_hierarchical_parallel_for(device, work_group_size, local_memory);
    sequential_for_work_group(num_work_groups, work_group_size, kernel);
}

} // namespace simsycl::detail
