#pragma once

#include "enums.hh"
#include "forward.hh"
#include "range.hh"

#include <memory>
#include <vector>

namespace simsycl::detail {

sycl::handler make_handler();

} // namespace simsycl::detail

namespace simsycl::sycl {

class handler {
  public:
    handler(const handler &) = delete;
    handler &operator=(const handler &) = delete;

    template <typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget,
        access::placeholder IsPlaceholder>
    void require(accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder> acc);

    void depends_on(event dep_event);

    void depends_on(const std::vector<event> &dep_events);

    //----- Backend interoperability interface

    template <typename T>
    void set_arg(int arg_index, T &&arg);

    template <typename... Ts>
    void set_args(Ts &&...args);

    //------ Kernel dispatch API

    // Note: In all kernel dispatch functions, the template parameter "typename KernelName" is optional.

    template <typename KernelName, typename KernelType>
    void single_task(const KernelType &kernel_func);

    // Parameter pack acts as-if: Reductions&&... reductions, const KernelType &kernel_func
    template <typename KernelName, int Dimensions, typename... Rest>
    void parallel_for(range<Dimensions> num_work_items, Rest &&...rest) {
        // TODO
    }

    template <typename KernelName, typename KernelType, int Dimensions>
    [[deprecated("Deprecated in SYCL 2020")]] void parallel_for(
        range<Dimensions> num_work_items, id<Dimensions> work_item_offset, const KernelType &kernel_func);

    // Parameter pack acts as-if: Reductions&&... reductions, const KernelType &kernel_func
    template <typename KernelName, int Dimensions, typename... Rest>
    void parallel_for(nd_range<Dimensions> execution_range, Rest &&...rest);

    template <typename KernelName, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(range<Dimensions> num_work_groups, const WorkgroupFunctionType &kernel_func);

    template <typename KernelName, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(
        range<Dimensions> num_work_groups, range<Dimensions> work_group_size, const WorkgroupFunctionType &kernel_func);

    void single_task(const kernel &kernel_object);

    template <int Dimensions>
    void parallel_for(range<Dimensions> num_work_items, const kernel &kernel_object);

    template <int Dimensions>
    void parallel_for(nd_range<Dimensions> nd_range, const kernel &kernel_object);

    //------ USM functions

    void memcpy(void *dest, const void *src, size_t num_bytes);

    template <typename T>
    void copy(const T *src, T *dest, size_t count);

    void memset(void *ptr, int value, size_t num_bytes);

    template <typename T>
    void fill(void *ptr, const T &pattern, size_t count);

    void prefetch(void *ptr, size_t num_bytes);

    void mem_advise(void *ptr, size_t num_bytes, int advice);

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
