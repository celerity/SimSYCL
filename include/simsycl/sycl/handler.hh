#pragma once

#include <any>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "accessor.hh"
#include "enums.hh"
#include "event.hh"
#include "forward.hh"
#include "id.hh"
#include "interop_handle.hh"
#include "item.hh"
#include "kernel.hh"
#include "nd_range.hh"
#include "range.hh"

#include "../detail/nd_memory.hh"
#include "../detail/parallel_for.hh"


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
    void set_arg(int arg_index, T &&arg) {
        (void)arg_index;
        (void)arg;
        SIMSYCL_CHECK(!"SimSYCL does not have built-in or backend interop kernels, calling set_arg is undefined");
    }

    template<typename... Ts>
    void set_args(Ts &&...args) {
        ((void)args, ...);
        SIMSYCL_CHECK(!"SimSYCL does not have built-in or backend interop kernels, calling set_args is undefined");
    }

    //------ Host tasks

    template<typename T>
    void host_task(T &&host_task_callable) {
        // TODO pass interop_handle if possible
        if constexpr(std::is_invocable_v<T, interop_handle>) {
            host_task_callable(detail::make_interop_handle());
        } else {
            host_task_callable();
        }
    }

    //------ Kernel dispatch API

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType>
    void single_task(const KernelType &kernel_func) {
        detail::execute_single_task<KernelName>(kernel_handler(this), kernel_func);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest>
        requires(sizeof...(Rest) > 0)
    void parallel_for(size_t num_work_items, Rest &&...rest) {
        detail::parallel_for<KernelName>(range<1>(num_work_items), kernel_handler(this), std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest>
        requires(sizeof...(Rest) > 0 && Dimensions > 0)
    void parallel_for(range<Dimensions> num_work_items, Rest &&...rest) {
        detail::parallel_for<KernelName>(num_work_items, kernel_handler(this), std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType, int Dimensions>
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL void parallel_for(
        range<Dimensions> num_work_items, id<Dimensions> work_item_offset, KernelType &&kernel_func) {
        detail::parallel_for<KernelName>(num_work_items, work_item_offset, kernel_handler(this), kernel_func);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dimensions, typename... Rest>
        requires(sizeof...(Rest) > 0)
    void parallel_for(nd_range<Dimensions> execution_range, Rest &&...rest) {
        detail::parallel_for<KernelName>(
            m_device, execution_range, m_local_memory, kernel_handler(this), std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(range<Dimensions> num_work_groups, const WorkgroupFunctionType &kernel_func) {
        detail::parallel_for_work_group<KernelName>(
            m_device, num_work_groups, {}, m_local_memory, kernel_handler(this), kernel_func);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename WorkgroupFunctionType, int Dimensions>
    void parallel_for_work_group(range<Dimensions> num_work_groups, range<Dimensions> work_group_size,
        const WorkgroupFunctionType &kernel_func) {
        detail::parallel_for_work_group<KernelName>(
            m_device, num_work_groups, {work_group_size}, m_local_memory, kernel_handler(this), kernel_func);
    }

    void single_task(const kernel &kernel_object) {
        (void)kernel_object;
        throw sycl::exception(sycl::errc::invalid, "SYCL does not allow invoking user kernels via kernel object APIs");
    }

    template<int Dimensions>
    void parallel_for(range<Dimensions> num_work_items, const kernel &kernel_object) {
        (void)num_work_items;
        (void)kernel_object;
        throw sycl::exception(sycl::errc::invalid, "SYCL does not allow invoking user kernels via kernel object APIs");
    }

    template<int Dimensions>
    void parallel_for(nd_range<Dimensions> nd_range, const kernel &kernel_object) {
        (void)nd_range;
        (void)kernel_object;
        throw sycl::exception(sycl::errc::invalid, "SYCL does not allow invoking user kernels via kernel object APIs");
    }

    //------ USM functions

    void memcpy(void *dest, const void *src, size_t num_bytes) { ::memcpy(dest, src, num_bytes); }

    template<typename T>
    void copy(const T *src, T *dest, size_t count) {
        std::copy_n(src, count, dest);
    }

    void memset(void *ptr, int value, size_t num_bytes) { ::memset(ptr, value, num_bytes); }

    template<typename T>
    void fill(void *ptr, const T &pattern, size_t count) {
        std::fill_n(static_cast<T *>(ptr), count, pattern);
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
    void update_host(accessor<T, Dim, Mode, Tgt, IsPlaceholder> acc) {
        acc.update_host();
    }

    template<typename T, int Dim, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    void fill(accessor<T, Dim, Mode, Tgt, IsPlaceholder> dest, const T &src) {
        parallel_for(dest.get_range(), dest.get_offset(), [&](item<Dim> item) { dest[item] = src; });
    }

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

    void use_kernel_bundle(const kernel_bundle<bundle_state::executable> &exec_bundle);

    template<auto &SpecName>
    void set_specialization_constant(typename std::remove_reference_t<decltype(SpecName)>::value_type value) {
        static_assert(detail::is_specialization_id_v<std::remove_cvref_t<decltype(SpecName)>>);
        if(auto existing = find_specialization_constant(this, &SpecName)) {
            *existing = value;
        } else {
            m_specialization_constants.emplace_back(&SpecName, value);
        }
    }

    template<auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant() const {
        static_assert(detail::is_specialization_id_v<std::remove_cvref_t<decltype(SpecName)>>);
        if(auto existing = find_specialization_constant(this, &SpecName)) {
            return std::any_cast<typename std::remove_reference_t<decltype(SpecName)>::value_type>(*existing);
        }
        return detail::get_specialization_default(SpecName);
    }

  private:
    friend handler simsycl::detail::make_handler(const sycl::device &device);
    friend void **simsycl::detail::require_local_memory(handler &cgh, size_t size, size_t align);

    device m_device;
    std::vector<detail::local_memory_requirement> m_local_memory;
    std::vector<std::pair<const void *, std::any>> m_specialization_constants;

    explicit handler(const device &device) : m_device(device) {}

    static auto find_specialization_constant(auto self, const void *spec_id)
        -> decltype(&self->m_specialization_constants[0].second) {
        if(const auto it = std::find_if(self->m_specialization_constants.begin(),
               self->m_specialization_constants.end(), [&](const auto &pair) { return pair.first == spec_id; });
            it != self->m_specialization_constants.end()) {
            return &it->second;
        }
        return nullptr;
    }
};

template<auto &SpecName>
typename std::remove_reference_t<decltype(SpecName)>::value_type kernel_handler::get_specialization_constant() {
    return m_cgh->get_specialization_constant<SpecName>();
}

} // namespace simsycl::sycl

namespace simsycl::detail {

inline sycl::handler make_handler(const sycl::device &device) { return sycl::handler(device); }

inline void **require_local_memory(sycl::handler &cgh, const size_t size, const size_t align) {
    cgh.m_local_memory.push_back(local_memory_requirement{std::make_unique<void *>(), size, align});
    return cgh.m_local_memory.back().ptr.get();
}

} // namespace simsycl::detail
