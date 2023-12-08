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

#include "simsycl/detail/utils.hh"

namespace simsycl::detail {

sycl::handler make_handler();

template <typename Func, typename... Params>
void sequential_for(const sycl::range<1> &range, const sycl::id<1> &offset, Func &&func, Params &&...args) {
    sycl::id<1> id;
    for(id[0] = offset[0]; id[0] < offset[0] + range[0]; ++id[0]) { //
        func(make_item(id, range, offset), std::forward<Params>(args)...);
    }
}

template <typename Func, typename... Params>
void sequential_for(const sycl::range<2> &range, const sycl::id<2> &offset, Func &&func, Params &&...args) {
    sycl::id<2> id;
    for(id[0] = offset[0]; id[0] < offset[0] + range[0]; ++id[0]) {
        for(id[1] = offset[1]; id[1] < offset[1] + range[1]; ++id[1]) { //
            func(make_item(id, range, offset), std::forward<Params>(args)...);
        }
    }
}

template <typename Func, typename... Params>
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


template <typename Func, typename... Params>
void nd_for(const sycl::nd_range<1> &range, Func &&func, Params &&...args) {
    sycl::id<1> global_id;
    std::vector<detail::nd_item_impl> nd_item_impls(range.get_global_range()[0]);
    std::vector<boost::context::continuation> continuations;

    std::vector<detail::group_impl> group_impls(range.get_group_range()[0]);
    auto sub_groups_per_group = detail::div_ceil(range.get_local_range()[0], config::max_sub_group_size);
    std::vector<detail::sub_group_impl> sub_group_impls(range.get_group_range()[0] * sub_groups_per_group);

    // build the work items, groups, subgroups and continuations
    for(global_id[0] = 0; global_id[0] < range.get_global_range()[0]; ++global_id[0]) {
        auto *nd_item_impl_ptr = &nd_item_impls[global_id[0]];
        auto global_item = detail::make_item(global_id, range.get_global_range());
        auto local_item = detail::make_item(global_id % range.get_local_range()[0], range.get_local_range());

        sycl::id<1> group_id{global_id[0] / range.get_local_range()[0]};
        sycl::item<1, false> group_item = detail::make_item(group_id, range.get_group_range());
        auto *group_impl_ptr = &group_impls[group_id[0]];
        group_impl_ptr->item_impls.push_back(nd_item_impl_ptr);
        auto group = detail::make_group(local_item, global_item, group_item, group_impl_ptr);

        sycl::id<1> sub_group_local_id{local_item[0] % config::max_sub_group_size};
        sycl::range<1> sub_group_local_range{config::max_sub_group_size};
        sycl::id<1> sub_group_group_id{local_item[0] / config::max_sub_group_size};
        sycl::range<1> sub_group_group_range{sub_groups_per_group};
        auto *sub_group_impl_ptr = &sub_group_impls[group_id[0] * sub_groups_per_group + sub_group_group_id[0]];
        sub_group_impl_ptr->item_impls.push_back(nd_item_impl_ptr);
        auto sub_group = detail::make_sub_group(
            sub_group_local_id, sub_group_local_range, sub_group_group_id, sub_group_group_range, sub_group_impl_ptr);

        auto nd_item = detail::make_nd_item<1>(global_item, local_item, group, sub_group, nd_item_impl_ptr);

        continuations.push_back(boost::context::callcc(
            [func, nd_item, global_id, &nd_item_impls, &args...](boost::context::continuation &&cont) {
                auto &impl = nd_item_impls[global_id[0]];
                impl.continuation = &cont;
                impl.state = detail::nd_item_state::running;
                func(nd_item, std::forward<Params>(args)...);
                impl.state = nd_item_state::exit;
                return std::move(cont);
            }));
    }

    // run until all are complete (this does an extra loop)
    bool done = false;
    while(!done) {
        done = true;
        for(size_t i = 0; i < nd_item_impls.size(); ++i) {
            if(nd_item_impls[i].state != nd_item_state::exit) {
                done = false;
                continuations[i] = continuations[i].resume();
            }
        }
    }
}

template <int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_for(const sycl::range<Dimensions> &range, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const sycl::id<Dimensions> offset{};
    const auto &kernel_func = std::get<KernelIndex>(params);
    detail::sequential_for(range, offset, kernel_func, std::get<ReductionIndices>(params)...);
}

template <int Dimensions, typename ParamTuple, size_t... ReductionIndices, size_t KernelIndex>
void dispatch_for(const sycl::nd_range<Dimensions> &range, ParamTuple &&params,
    std::index_sequence<ReductionIndices...> /* reduction_indices */,
    std::index_sequence<KernelIndex> /* kernel_index */) {
    const auto &kernel_func = std::get<KernelIndex>(params);
    detail::nd_for(range, kernel_func, std::get<ReductionIndices>(params)...);
}

template <int Dimensions, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(sycl::range<Dimensions> num_work_items, Rest &&...rest) {
    dispatch_for(num_work_items, std::forward_as_tuple(std::forward<Rest>(rest)...),
        std::make_index_sequence<sizeof...(Rest) - 1>(), std::index_sequence<sizeof...(Rest) - 1>());
}

template <typename KernelType, int Dimensions>
void parallel_for(
    sycl::range<Dimensions> num_work_items, sycl::id<Dimensions> work_item_offset, KernelType &&kernel_func) {
    detail::sequential_for(num_work_items, work_item_offset, kernel_func);
}

template <typename KernelName = unnamed_kernel, int Dimensions, typename... Rest,
    std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
void parallel_for(sycl::nd_range<Dimensions> execution_range, Rest &&...rest) {
    detail::dispatch_for(execution_range, std::forward_as_tuple(std::forward<Rest>(rest)...),
        std::make_index_sequence<sizeof...(Rest) - 1>(), std::index_sequence<sizeof...(Rest) - 1>());
}

} // namespace simsycl::detail

namespace simsycl::sycl {

class handler {
  public:
    handler(const handler &) = delete;
    handler &operator=(const handler &) = delete;

    template <typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget,
        access::placeholder IsPlaceholder>
    void require(accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder> acc);

    void depends_on(event dep_event) { (void)dep_event; }

    void depends_on(const std::vector<event> &dep_events) { (void)dep_events; }

    //----- Backend interoperability interface

    template <typename T>
    void set_arg(int arg_index, T &&arg);

    template <typename... Ts>
    void set_args(Ts &&...args);

    //------ Kernel dispatch API

    template <typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType>
    void single_task(const KernelType &kernel_func) {
        kernel_func();
    }

    template <typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    void parallel_for(range<Dimensions> num_work_items, Rest &&...rest) {
        simsycl::detail::parallel_for(num_work_items, std::forward<Rest>(rest)...);
    }

    template <typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType, int Dimensions>
    [[deprecated("Deprecated in SYCL 2020")]] void parallel_for(
        range<Dimensions> num_work_items, id<Dimensions> work_item_offset, KernelType &&kernel_func) {
        simsycl::detail::parallel_for(num_work_items, work_item_offset, kernel_func);
    }

    template <typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    void parallel_for(nd_range<Dimensions> execution_range, Rest &&...rest) {
        simsycl::detail::parallel_for(execution_range, std::forward<Rest>(rest)...);
    }

    template <typename KernelName = simsycl::detail::unnamed_kernel, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(range<Dimensions> num_work_groups, const WorkgroupFunctionType &kernel_func);

    template <typename KernelName = simsycl::detail::unnamed_kernel, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(
        range<Dimensions> num_work_groups, range<Dimensions> work_group_size, const WorkgroupFunctionType &kernel_func);

    void single_task(const kernel &kernel_object);

    template <int Dimensions>
    void parallel_for(range<Dimensions> num_work_items, const kernel &kernel_object);

    template <int Dimensions>
    void parallel_for(nd_range<Dimensions> nd_range, const kernel &kernel_object);

    //------ USM functions

    void memcpy(void *dest, const void *src, size_t num_bytes) { ::memcpy(dest, src, num_bytes); }

    template <typename T>
    void copy(const T *src, T *dest, size_t count) {
        std::copy_n(src, count, dest);
    }

    void memset(void *ptr, int value, size_t num_bytes) { ::memset(ptr, value, num_bytes); }

    template <typename T>
    void fill(void *ptr, const T &pattern, size_t count) {
        std::fill_n(ptr, count, pattern);
    }

    void prefetch(void * /* ptr */, size_t /* num_bytes */) {}

    void mem_advise(void * /* ptr */, size_t /* num_bytes */, int /* advice */) {}

    //------ Explicit memory operation APIs
    //
    template <typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src, std::shared_ptr<DestT> dest);

    template <typename SrcT, typename DestT, int DestDim, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    void copy(std::shared_ptr<SrcT> src, accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest);

    template <typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, IsPlaceholder> src, DestT *dest);

    template <typename SrcT, typename DestT, int DestDim, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    void copy(const SrcT *src, accessor<DestT, DestDim, DestMode, DestTgt, IsPlaceholder> dest);

    template <typename SrcT, int SrcDim, access_mode SrcMode, target SrcTgt, access::placeholder SrcIsPlaceholder,
        typename DestT, int DestDim, access_mode DestMode, target DestTgt, access::placeholder DestIsPlaceholder>
    void copy(accessor<SrcT, SrcDim, SrcMode, SrcTgt, SrcIsPlaceholder> src,
        accessor<DestT, DestDim, DestMode, DestTgt, DestIsPlaceholder> dest);

    template <typename T, int Dim, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    void update_host(accessor<T, Dim, Mode, Tgt, IsPlaceholder> acc);

    template <typename T, int Dim, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    void fill(accessor<T, Dim, Mode, Tgt, IsPlaceholder> dest, const T &src);

    void use_kernel_bundle(const kernel_bundle<bundle_state::executable> &exec_bundle);

    template <auto &SpecName>
    void set_specialization_constant(typename std::remove_reference_t<decltype(SpecName)>::value_type value);

    template <auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant();

  private:
    friend handler simsycl::detail::make_handler();

    handler() = default;
};

} // namespace simsycl::sycl

namespace simsycl::detail {

sycl::handler make_handler() { return {}; }

} // namespace simsycl::detail
