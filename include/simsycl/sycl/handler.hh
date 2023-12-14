#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "enums.hh"
#include "event.hh"
#include "forward.hh"
#include "group.hh"
#include "id.hh"
#include "item.hh"
#include "nd_item.hh"
#include "nd_range.hh"
#include "range.hh"

#include "simsycl/detail/allocation.hh"
#include "simsycl/detail/utils.hh"


namespace simsycl::detail {

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


template<int Dimensions, typename Func, typename... Params>
void dispatch_for_nd_range(const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, Func &&func, Params &&...args) {
    const auto &global_range = range.get_global_range();
    const auto global_linear_range = global_range.size();
    if(global_linear_range == 0) return;
    const auto &group_range = range.get_group_range();
    const auto group_linear_range = group_range.size();
    assert(group_linear_range > 0);
    const auto &local_range = range.get_local_range();
    const auto local_linear_range = local_range.size();
    assert(local_linear_range > 0);
    const auto sub_group_local_linear_range = config::max_sub_group_size;
    const auto sub_group_local_range = sycl::range<1>(sub_group_local_linear_range);
    assert(sub_group_local_linear_range > 0);
    const auto sub_group_linear_range_in_group = detail::div_ceil(local_linear_range, sub_group_local_linear_range);
    const sycl::range<1> sub_group_range_in_group{sub_group_linear_range_in_group};
    assert(sub_group_linear_range_in_group > 0);

    std::vector<detail::group_impl> group_impls(group_linear_range);
    std::vector<detail::sub_group_impl> sub_group_impls(group_linear_range * sub_group_linear_range_in_group);
    std::vector<detail::nd_item_impl> nd_item_impls(global_linear_range);
    std::vector<boost::context::continuation> continuations;

    for(size_t group_linear_id = 0; group_linear_id < group_linear_range; ++group_linear_id) {
        group_impls[group_linear_id].local_memory_allocations.resize(local_memory.size());
        for(size_t i = 0; i < local_memory.size(); ++i) {
            group_impls[group_linear_id].local_memory_allocations[i]
                = allocation(local_memory[i].size, local_memory[i].align);
        }
    }

    // build the work items, groups, subgroups and continuations
    for(size_t global_linear_id = 0; global_linear_id < range.get_global_range().size(); ++global_linear_id) {
        const auto global_id = linear_index_to_id(global_range, global_linear_id);
        const auto local_id = global_id % sycl::id(local_range);
        const auto group_id = global_id / sycl::id(local_range);

        const auto local_linear_id = get_linear_index(local_range, local_id);
        const auto group_linear_id = get_linear_index(group_range, group_id);

        const auto sub_group_linear_id_in_group = local_linear_id / sub_group_local_linear_range;
        const auto thread_linear_id_in_sub_group = local_linear_id % sub_group_local_linear_range;
        const auto sub_group_id_in_group = sycl::id<1>(sub_group_linear_id_in_group);
        const auto thread_id_in_sub_group = sycl::id<1>(thread_linear_id_in_sub_group);

        const auto global_item = detail::make_item(global_id, range.get_global_range());
        const auto local_item = detail::make_item(local_id, range.get_local_range());
        const auto group_item = detail::make_item(group_id, range.get_group_range());

        const auto nd_item_impl_ptr = &nd_item_impls[global_linear_id];
        const auto group_impl_ptr = &group_impls[group_linear_id];
        group_impl_ptr->item_impls.push_back(nd_item_impl_ptr);
        nd_item_impl_ptr->group = group_impl_ptr;
        const auto group = detail::make_group(local_item, global_item, group_item, group_impl_ptr);

        const auto sub_group_impl_ptr
            = &sub_group_impls[group_linear_id * sub_group_linear_range_in_group + sub_group_linear_id_in_group];
        sub_group_impl_ptr->item_impls.push_back(nd_item_impl_ptr);
        const auto sub_group = detail::make_sub_group(thread_id_in_sub_group, sub_group_local_range,
            sub_group_id_in_group, sub_group_range_in_group, sub_group_impl_ptr);

        const auto nd_item = detail::make_nd_item(global_item, local_item, group, sub_group, nd_item_impl_ptr);

        // adjust local memory pointers before spawning each fiber
        for(size_t i = 0; i < local_memory.size(); ++i) {
            *local_memory[i].ptr = group_impls[group_linear_id].local_memory_allocations[i].get();
        }

        continuations.push_back(
            boost::context::callcc([func, nd_item, nd_item_impl_ptr, &args...](boost::context::continuation &&cont) {
                nd_item_impl_ptr->continuation = &cont;
                nd_item_impl_ptr->state = detail::nd_item_state::running;
                func(nd_item, std::forward<Params>(args)...);
                nd_item_impl_ptr->state = nd_item_state::exit;
                return std::move(cont);
            }));
    }

    // run until all are complete (this does an extra loop)
    bool done = false;
    while(!done) {
        done = true;
        for(size_t global_linear_id = 0; global_linear_id < global_linear_range; ++global_linear_id) {
            if(nd_item_impls[global_linear_id].state == nd_item_state::exit) continue;
            done = false;

            // adjust local memory pointers before switching fibers
            const auto group = nd_item_impls[global_linear_id].group;
            for(size_t i = 0; i < local_memory.size(); ++i) {
                *local_memory[i].ptr = group->local_memory_allocations[i].get();
            }

            continuations[global_linear_id] = continuations[global_linear_id].resume();
        }
    }
}

template<int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_for(const sycl::range<Dimensions> &range, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const sycl::id<Dimensions> offset{};
    const auto &kernel_func = std::get<KernelIndex>(params);
    detail::sequential_for(range, offset, kernel_func, std::get<ReductionIndices>(params)...);
}

template<int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_for(const sycl::nd_range<Dimensions> &range, const std::vector<local_memory_requirement> &local_memory,
    ParamTuple &&params, std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const auto &kernel_func = std::get<KernelIndex>(params);
    detail::dispatch_for_nd_range(range, local_memory, kernel_func, std::get<ReductionIndices>(params)...);
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
void parallel_for(sycl::nd_range<Dimensions> execution_range, const std::vector<local_memory_requirement> &local_memory,
    Rest &&...rest) {
    detail::dispatch_for(execution_range, local_memory, std::forward_as_tuple(std::forward<Rest>(rest)...),
        std::make_index_sequence<sizeof...(Rest) - 1>(), std::index_sequence<sizeof...(Rest) - 1>());
}

} // namespace simsycl::detail


namespace simsycl::sycl {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // placeholder

class handler {
  public:
    handler(const handler &) = delete;
    handler &operator=(const handler &) = delete;

    template<typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget,
        access::placeholder IsPlaceholder>
    void require(accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder> acc);

    void depends_on(event dep_event) { (void)dep_event; }

    void depends_on(const std::vector<event> &dep_events) { (void)dep_events; }

    //----- Backend interoperability interface

    template<typename T>
    void set_arg(int arg_index, T &&arg);

    template<typename... Ts>
    void set_args(Ts &&...args);

    //------ Host tasks

    template<typename T>
    void host_task(T &&host_task_callable) {
        // TODO pass interop_handle if possible
        host_task_callable();
    }

    //------ Kernel dispatch API

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType>
    void single_task(const KernelType &kernel_func) {
        kernel_func();
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    void parallel_for(range<Dimensions> num_work_items, Rest &&...rest) {
        simsycl::detail::parallel_for(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType, int Dimensions>
    [[deprecated("Deprecated in SYCL 2020")]] void parallel_for(
        range<Dimensions> num_work_items, id<Dimensions> work_item_offset, KernelType &&kernel_func) {
        simsycl::detail::parallel_for(num_work_items, work_item_offset, kernel_func);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    void parallel_for(nd_range<Dimensions> execution_range, Rest &&...rest) {
        simsycl::detail::parallel_for(execution_range, m_local_memory, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(range<Dimensions> num_work_groups, const WorkgroupFunctionType &kernel_func);

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(
        range<Dimensions> num_work_groups, range<Dimensions> work_group_size, const WorkgroupFunctionType &kernel_func);

    void single_task(const kernel &kernel_object);

    template<int Dimensions>
    void parallel_for(range<Dimensions> num_work_items, const kernel &kernel_object);

    template<int Dimensions>
    void parallel_for(nd_range<Dimensions> nd_range, const kernel &kernel_object);

    //------ USM functions

    void memcpy(void *dest, const void *src, size_t num_bytes) { ::memcpy(dest, src, num_bytes); }

    template<typename T>
    void copy(const T *src, T *dest, size_t count) {
        std::copy_n(src, count, dest);
    }

    void memset(void *ptr, int value, size_t num_bytes) { ::memset(ptr, value, num_bytes); }

    template<typename T>
    void fill(void *ptr, const T &pattern, size_t count) {
        std::fill_n(ptr, count, pattern);
    }

    void prefetch(void * /* ptr */, size_t /* num_bytes */) {}

    void mem_advise(void * /* ptr */, size_t /* num_bytes */, int /* advice */) {}

    //------ Explicit memory operation APIs

    template<typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src, std::shared_ptr<DestT> dest);

    template<typename SrcT, typename DestT, int DestDim, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    void copy(std::shared_ptr<SrcT> src, accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest);

    template<typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src, DestT *dest);

    template<typename SrcT, typename DestT, int DestDim, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    void copy(const SrcT *src, accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest);

    template<typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder SrcIsPlaceholder,
        typename DestT, int DestDim, access_mode DestMode, target DestTgt, access::placeholder DestIsPlaceholder>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, SrcIsPlaceholder> src,
        accessor<DestT, DestDim, DestMode, DestTgt, DestIsPlaceholder> dest);

    template<typename T, int Dim, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    void update_host(accessor<T, Dim, Mode, Tgt, IsPlaceholder> acc);

    template<typename T, int Dim, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    void fill(accessor<T, Dim, Mode, Tgt, IsPlaceholder> dest, const T &src);

#pragma GCC diagnostic pop

    void use_kernel_bundle(const kernel_bundle<bundle_state::executable> &exec_bundle);

    template<auto &SpecName>
    void set_specialization_constant(typename std::remove_reference_t<decltype(SpecName)>::value_type value);

    template<auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant();

  private:
    friend handler simsycl::detail::make_handler();
    friend void **simsycl::detail::require_local_memory(handler &cgh, size_t size, size_t align);

    std::vector<detail::local_memory_requirement> m_local_memory;

    handler() = default;
};

#pragma GCC diagnostic pop

} // namespace simsycl::sycl

namespace simsycl::detail {

inline sycl::handler make_handler() { return {}; }

inline void **require_local_memory(sycl::handler &cgh, const size_t size, const size_t align) {
    cgh.m_local_memory.push_back(local_memory_requirement{std::make_unique<void *>(), size, align});
    return cgh.m_local_memory.back().ptr.get();
}

} // namespace simsycl::detail
