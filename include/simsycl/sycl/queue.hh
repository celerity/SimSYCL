#pragma once

#include "event.hh"
#include "exception.hh"
#include "handler.hh"
#include "info.hh"
#include "property.hh"

#include "../detail/reference_type.hh"


namespace simsycl::sycl::property::queue {

class enable_profiling {};
class in_order {};

} // namespace simsycl::sycl::property::queue

namespace simsycl::sycl {

template <>
struct is_property<property::queue::enable_profiling> : std::true_type {};
template <>
struct is_property_of<property::queue::enable_profiling, queue> : std::true_type {};

template <>
struct is_property<property::queue::in_order> : std::true_type {};
template <>
struct is_property_of<property::queue::in_order, queue> : std::true_type {};

} // namespace simsycl::sycl

namespace simsycl::detail {

struct queue_state {
    sycl::async_handler async_handler;

    queue_state() : async_handler([](sycl::exception_list) {}) {}
    queue_state(sycl::async_handler async_handler) : async_handler(std::move(async_handler)) {}
};

} // namespace simsycl::detail

namespace simsycl::sycl {

class queue : public detail::reference_type<queue, detail::queue_state>, public simsycl::detail::property_interface {
  private:
    using reference_type = detail::reference_type<queue, detail::queue_state>;
    using property_compatibility = simsycl::detail::property_compatibility_with<queue,
        property::queue::enable_profiling, property::queue::in_order>;

  public:
    explicit queue(const property_list &prop_list = {})
        : reference_type(std::in_place), property_interface(prop_list, property_compatibility()) {}

    explicit queue(const async_handler &async_handler, const property_list &prop_list = {})
        : reference_type(std::in_place, async_handler), property_interface(prop_list, property_compatibility()) {}

    template <typename DeviceSelector>
    explicit queue(const DeviceSelector &device_selector, const property_list &prop_list = {});

    template <typename DeviceSelector>
    explicit queue(
        const DeviceSelector &device_selector, const async_handler &async_handler, const property_list &prop_list = {});

    explicit queue(const device &sycl_device, const property_list &prop_list = {});

    explicit queue(const device &sycl_device, const async_handler &async_handler, const property_list &prop_list = {});

    template <typename DeviceSelector>
    explicit queue(
        const context &sycl_context, const DeviceSelector &device_selector, const property_list &prop_list = {});

    template <typename DeviceSelector>
    explicit queue(const context &sycl_context, const DeviceSelector &device_selector,
        const async_handler &async_handler, const property_list &prop_list = {});

    explicit queue(const context &sycl_context, const device &sycl_device, const property_list &prop_list = {});

    explicit queue(const context &sycl_context, const device &sycl_device, const async_handler &async_handler,
        const property_list &prop_list = {});

    backend get_backend() const noexcept;

    context get_context() const;

    device get_device() const;

    bool is_in_order() const { return has_property<property::queue::in_order>(); }

    template <typename Param>
    typename Param::return_type get_info() const {
        return {};
    }

    template <typename Param>
    typename Param::return_type get_backend_info() const {
        return {};
    }

    template <typename T>
    event submit(T cgf) {
        auto status = detail::execution_status::submit();
        auto cgh = simsycl::detail::make_handler();
        status.start();
        cgf(cgh);
        return status.end();
    }

    template <typename T>
    event submit(T cgf, const queue & /* secondary_queue */) {
        // TODO can the secondary queue be interesting for some device configurations?
        return submit(cgf);
    }

    void wait() {}
    void wait_and_throw() {}
    void throw_asynchronous() {}

    /* -- convenience shortcuts -- */

    template <typename KernelName, typename KernelType>
    event single_task(const KernelType &kernel_func) {
        auto status = detail::execution_status::submit_and_start();
        kernel_func();
        return status.end();
    }

    template <typename KernelName, typename KernelType>
    event single_task(event /* dep_event */, const KernelType &kernel_func) {
        auto status = detail::execution_status::submit_and_start();
        kernel_func();
        return status.end();
    }

    template <typename KernelName, typename KernelType>
    event single_task(const std::vector<event> & /* dep_events */, const KernelType &kernel_func) {
        auto status = detail::execution_status::submit_and_start();
        kernel_func();
        return status.end();
    }

    template <typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<Dims> num_work_items, Rest &&...rest) {
        auto status = detail::execution_status::submit_and_start();
        simsycl::detail::parallel_for(num_work_items, std::forward<Rest>(rest)...);
        return status.end();
    }

    template <typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<Dims> num_work_items, event /* dep_event */, Rest &&...rest) {
        auto status = detail::execution_status::submit_and_start();
        simsycl::detail::parallel_for(num_work_items, std::forward<Rest>(rest)...);
        return status.end();
    }

    template <typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<Dims> num_work_items, const std::vector<event> & /* dep_events */, Rest &&...rest) {
        auto status = detail::execution_status::submit_and_start();
        simsycl::detail::parallel_for(num_work_items, std::forward<Rest>(rest)...);
        return status.end();
    }

    template <typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(nd_range<Dims> execution_range, Rest &&...rest) {
        auto status = detail::execution_status::submit_and_start();
        simsycl::detail::parallel_for(execution_range, std::forward<Rest>(rest)...);
        return status.end();
    }

    template <typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(nd_range<Dims> execution_range, event /* dep_event */, Rest &&...rest) {
        auto status = detail::execution_status::submit_and_start();
        simsycl::detail::parallel_for(execution_range, std::forward<Rest>(rest)...);
        return status.end();
    }

    template <typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(nd_range<Dims> execution_range, const std::vector<event> & /* dep_events */, Rest &&...rest) {
        auto status = detail::execution_status::submit_and_start();
        simsycl::detail::parallel_for(execution_range, std::forward<Rest>(rest)...);
        return status.end();
    }

    /* -- USM functions -- */

    event memcpy(void *dest, const void *src, size_t num_bytes) {
        auto status = detail::execution_status::submit_and_start();
        ::memcpy(dest, src, num_bytes);
        return status.end();
    }

    event memcpy(void *dest, const void *src, size_t num_bytes, event /* dep_event */) {
        auto status = detail::execution_status::submit_and_start();
        ::memcpy(dest, src, num_bytes);
        return status.end();
    }

    event memcpy(void *dest, const void *src, size_t num_bytes, const std::vector<event> & /* dep_events */) {
        auto status = detail::execution_status::submit_and_start();
        ::memcpy(dest, src, num_bytes);
        return status.end();
    }

    template <typename T>
    event copy(const T *src, T *dest, size_t count) {
        auto status = detail::execution_status::submit_and_start();
        std::copy_n(src, count, dest);
        return status.end();
    }

    template <typename T>
    event copy(const T *src, T *dest, size_t count, event dep_event) {
        (void)(dep_event);
        auto status = detail::execution_status::submit_and_start();
        std::copy_n(src, count, dest);
        return status.end();
    }

    template <typename T>
    event copy(const T *src, T *dest, size_t count, const std::vector<event> &dep_events) {
        (void)(dep_events);
        auto status = detail::execution_status::submit_and_start();
        std::copy_n(src, count, dest);
        return status.end();
    }

    event memset(void *ptr, int value, size_t num_bytes) {
        auto status = detail::execution_status::submit_and_start();
        ::memset(ptr, value, num_bytes);
        return status.end();
    }

    event memset(void *ptr, int value, size_t num_bytes, event /* dep_event */) {
        auto status = detail::execution_status::submit_and_start();
        ::memset(ptr, value, num_bytes);
        return status.end();
    }

    event memset(void *ptr, int value, size_t num_bytes, const std::vector<event> & /* dep_events */) {
        auto status = detail::execution_status::submit_and_start();
        ::memset(ptr, value, num_bytes);
        return status.end();
    }

    template <typename T>
    event fill(void *ptr, const T &pattern, size_t count) {
        auto status = detail::execution_status::submit_and_start();
        std::fill_n(ptr, count, pattern);
        return status.end();
    }

    template <typename T>
    event fill(void *ptr, const T &pattern, size_t count, event /* dep_event */) {
        auto status = detail::execution_status::submit_and_start();
        std::fill_n(ptr, count, pattern);
        return status.end();
    }

    template <typename T>
    event fill(void *ptr, const T &pattern, size_t count, const std::vector<event> & /* dep_events */) {
        auto status = detail::execution_status::submit_and_start();
        std::fill_n(ptr, count, pattern);
        return status.end();
    }

    event prefetch(void * /* ptr */, size_t /* num_bytes */) { return detail::execution_status::instant(); }

    event prefetch(void * /* ptr */, size_t /* num_bytes */, event /* dep_event */) {
        return detail::execution_status::instant();
    }

    event prefetch(void * /* ptr */, size_t /* num_bytes */, const std::vector<event> & /* dep_events */) {
        return detail::execution_status::instant();
    }

    event mem_advise(void * /* ptr */, size_t /* num_bytes */, int /* advice */) {
        return detail::execution_status::instant();
    }

    event mem_advise(void * /* ptr */, size_t /* num_bytes */, int /* advice */, event /* dep_event */) {
        return detail::execution_status::instant();
    }

    event mem_advise(
        void * /* ptr */, size_t /* num_bytes */, int /* advice */, const std::vector<event> & /* dep_events */) {
        return detail::execution_status::instant();
    }

    /// Placeholder accessor shortcuts

    // Explicit copy functions

    template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src, std::shared_ptr<DestT> dest);

    template <typename SrcT, typename DestT, int DestDims, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    event copy(std::shared_ptr<SrcT> src, accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest);

    template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src, DestT *dest);

    template <typename SrcT, typename DestT, int DestDims, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    event copy(const SrcT *src, accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest);

    template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt, access::placeholder IsSrcPlaceholder,
        typename DestT, int DestDims, access_mode DestMode, target DestTgt, access::placeholder IsDestPlaceholder>
    event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsSrcPlaceholder> src,
        accessor<DestT, DestDims, DestMode, DestTgt, IsDestPlaceholder> dest);

    template <typename T, int Dims, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    event update_host(accessor<T, Dims, Mode, Tgt, IsPlaceholder> acc);

    template <typename T, int Dims, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    event fill(accessor<T, Dims, Mode, Tgt, IsPlaceholder> dest, const T &src);
};

} // namespace simsycl::sycl
