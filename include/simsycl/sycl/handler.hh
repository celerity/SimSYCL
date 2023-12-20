#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "accessor.hh"
#include "device.hh"
#include "enums.hh"
#include "event.hh"
#include "forward.hh"
#include "id.hh"
#include "item.hh"
#include "nd_item.hh"
#include "nd_range.hh"
#include "range.hh"


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
    const sycl::id<Dimensions> offset{};
    const auto &kernel_func = std::get<KernelIndex>(params);
    detail::sequential_for(range, offset, kernel_func, std::get<ReductionIndices>(params)...);
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


namespace simsycl::sycl {

// placeholder
SIMSYCL_START_IGNORING_DEPRECATIONS

class handler {
  public:
    handler(const handler &) = delete;
    handler &operator=(const handler &) = delete;

    template<typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget,
        access::placeholder IsPlaceholder>
    void require(accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder> acc) {
        acc.require();
    }

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

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest>
        requires(sizeof...(Rest) > 0)
    void parallel_for(range<Dimensions> num_work_items, Rest &&...rest) {
        simsycl::detail::parallel_for(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType, int Dimensions>
    [[deprecated("Deprecated in SYCL 2020")]] void parallel_for(
        range<Dimensions> num_work_items, id<Dimensions> work_item_offset, KernelType &&kernel_func) {
        simsycl::detail::parallel_for(num_work_items, work_item_offset, kernel_func);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest>
        requires(sizeof...(Rest) > 0)
    void parallel_for(nd_range<Dimensions> execution_range, Rest &&...rest) {
        simsycl::detail::parallel_for(m_device, execution_range, m_local_memory, std::forward<Rest>(rest)...);
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
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src, std::shared_ptr<DestT> dest) {
        copy(src, dest.get());
    }

    template<typename SrcT, typename DestT, int DestDim, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    void copy(std::shared_ptr<SrcT> src, accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest) {
        copy(src.get(), dest);
    }

    template<typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src, DestT *dest) {
        static_assert(sizeof(SrcT) == sizeof(DestT));
        detail::memcpy_strided_host(src.get_pointer(), dest, sizeof(SrcT), src.get_buffer_range(), src.get_offset(),
            src.get_range(), sycl::id<SrcDim>(), src.get_range());
    }

    template<typename SrcT, typename DestT, int DestDim, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    void copy(const SrcT *src, accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest) {
        static_assert(sizeof(SrcT) == sizeof(DestT));
        detail::memcpy_strided_host(src, dest.get_pointer(), sizeof(SrcT), dest.get_range(), sycl::id<DestDim>(),
            dest.get_buffer_range(), dest.get_offset(), dest.get_range());
    }

    template<typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder SrcIsPlaceholder,
        typename DestT, int DestDim, access_mode DestMode, target DestTgt, access::placeholder DestIsPlaceholder>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, SrcIsPlaceholder> src,
        accessor<DestT, DestDim, DestMode, DestTgt, DestIsPlaceholder> dest) {
        static_assert(sizeof(SrcT) == sizeof(DestT));
        static_assert(SrcDim == DestDim, "copy between different accessor dimensions not implemented");
        assert(src.get_range() == dest.get_range() && "copy between differently-ranged accessors not implemented");
        detail::memcpy_strided_host(src.get_pointer(), dest.get_pointer(), sizeof(SrcT), src.get_buffer_range(),
            src.get_offset(), dest.get_buffer_range(), dest.get_offset(), dest.get_range());
    }

    template<typename T, int Dim, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    void update_host(accessor<T, Dim, Mode, Tgt, IsPlaceholder> acc);

    template<typename T, int Dim, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    void fill(accessor<T, Dim, Mode, Tgt, IsPlaceholder> dest, const T &src) {
        detail::sequential_for(dest.get_range(), dest.get_offset(), [&](const item<Dim> &item) { dest[item] = src; });
    }

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

    void use_kernel_bundle(const kernel_bundle<bundle_state::executable> &exec_bundle);

    template<auto &SpecName>
    void set_specialization_constant(typename std::remove_reference_t<decltype(SpecName)>::value_type value);

    template<auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant();

  private:
    friend handler simsycl::detail::make_handler(const sycl::device &device);
    friend void **simsycl::detail::require_local_memory(handler &cgh, size_t size, size_t align);

    device m_device;
    std::vector<detail::local_memory_requirement> m_local_memory;

    explicit handler(const device &device) : m_device(device) {}
};

} // namespace simsycl::sycl

namespace simsycl::detail {

inline sycl::handler make_handler(const sycl::device &device) { return sycl::handler(device); }

inline void **require_local_memory(sycl::handler &cgh, const size_t size, const size_t align) {
    cgh.m_local_memory.push_back(local_memory_requirement{std::make_unique<void *>(), size, align});
    return cgh.m_local_memory.back().ptr.get();
}

} // namespace simsycl::detail
