#pragma once

#include <simsycl/config.hh>

#include "async_handler.hh"
#include "event.hh"
#include "handler.hh"
#include "property.hh"

#include "../detail/lock.hh"
#include "../detail/reference_type.hh"

#if SIMSYCL_ENABLE_SYCL_KHR_QUEUE_FLUSH
#define SYCL_KHR_QUEUE_FLUSH 1
#endif // SIMSYCL_ENABLE_SYCL_KHR_QUEUE_FLUSH

namespace simsycl::sycl::property::queue {

class enable_profiling {};
class in_order {};

} // namespace simsycl::sycl::property::queue

namespace simsycl::sycl {

template<>
struct is_property<property::queue::enable_profiling> : std::true_type {};
template<>
struct is_property_of<property::queue::enable_profiling, queue> : std::true_type {};

template<>
struct is_property<property::queue::in_order> : std::true_type {};
template<>
struct is_property_of<property::queue::in_order, queue> : std::true_type {};

} // namespace simsycl::sycl

namespace simsycl::detail {

struct queue_state;

} // namespace simsycl::detail

namespace simsycl::sycl {

class queue final : public detail::reference_type<queue, detail::queue_state>,
                    public simsycl::detail::property_interface {
  private:
    using reference_type = detail::reference_type<queue, detail::queue_state>;
    using property_compatibility = simsycl::detail::property_compatibility_with<queue,
        property::queue::enable_profiling, property::queue::in_order>;

  public:
    explicit queue(const property_list &prop_list = {})
        : queue(internal, default_selector_v, async_handler{}, prop_list) {}

    explicit queue(const async_handler &async_handler, const property_list &prop_list = {})
        : queue(internal, default_selector_v, async_handler, prop_list) {}

    template<DeviceSelector Selector>
    explicit queue(const Selector &device_selector, const property_list &prop_list = {})
        : queue(internal, detail::device_selector(device_selector), async_handler{}, prop_list) {}

    template<DeviceSelector Selector>
    explicit queue(
        const Selector &device_selector, const async_handler &async_handler, const property_list &prop_list = {})
        : queue(internal, detail::device_selector(device_selector), async_handler, prop_list) {}

    explicit queue(const device &sycl_device, const property_list &prop_list = {})
        : queue(internal, sycl_device, async_handler{}, prop_list) {}

    explicit queue(const device &sycl_device, const async_handler &async_handler, const property_list &prop_list = {})
        : queue(internal, sycl_device, async_handler, prop_list) {}

    template<DeviceSelector Selector>
    explicit queue(const context &sycl_context, const Selector &device_selector, const property_list &prop_list = {})
        : queue(internal, sycl_context, detail::device_selector(device_selector), async_handler{}, prop_list) {}

    template<DeviceSelector Selector>
    explicit queue(const context &sycl_context, const Selector &device_selector, const async_handler &async_handler,
        const property_list &prop_list = {})
        : queue(internal, sycl_context, detail::device_selector(device_selector), async_handler, prop_list) {}

    explicit queue(const context &sycl_context, const device &sycl_device, const property_list &prop_list = {})
        : queue(internal, sycl_context, sycl_device, async_handler{}, prop_list) {}

    explicit queue(const context &sycl_context, const device &sycl_device, const async_handler &async_handler,
        const property_list &prop_list = {})
        : queue(internal, sycl_context, sycl_device, async_handler, prop_list) {}

    backend get_backend() const noexcept;

    context get_context() const;

    device get_device() const;

#if SIMSYCL_ENABLE_SYCL_KHR_QUEUE_FLUSH
    void khr_flush() const { /* This is a no-op in SimSYCL due to its synchronous nature */ }
#endif // SIMSYCL_ENABLE_SYCL_KHR_QUEUE_FLUSH

    bool is_in_order() const { return has_property<property::queue::in_order>(); }

    template<typename Param>
    typename Param::return_type get_info() const;

    template<typename Param>
    typename Param::return_type get_backend_info() const;

    template<typename T>
    event submit(T cgf) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        auto cgh = simsycl::detail::make_handler(get_device());
        status.start();
        cgf(cgh);
        return status.end();
    }

    template<typename T>
    event submit(T cgf, const queue &secondary_queue) {
        // TODO can the secondary queue be interesting for some device configurations?
        (void)secondary_queue;
        return submit(cgf);
    }

    void wait() {}
    void wait_and_throw() {}
    void throw_asynchronous() {}

    /* -- convenience shortcuts -- */

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelFunc>
    event single_task(const KernelFunc &kernel_func) {
        return submit([&](handler &cgh) { cgh.single_task<KernelName>(kernel_func); });
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType>
    event single_task(event dep_event, const KernelType &kernel_func) {
        (void)dep_event;
        return submit([&](handler &cgh) { cgh.single_task<KernelName>(kernel_func); });
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename KernelType>
    event single_task(const std::vector<event> &dep_events, const KernelType &kernel_func) {
        (void)dep_events;
        return submit([&](handler &cgh) { cgh.single_task<KernelName>(kernel_func); });
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(size_t num_work_items, Rest &&...rest) {
        return simple_parallel_for<KernelName>(range(num_work_items), std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<1> num_work_items, Rest &&...rest) {
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<2> num_work_items, Rest &&...rest) {
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<3> num_work_items, Rest &&...rest) {
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(size_t num_work_items, event dep_event, Rest &&...rest) {
        (void)dep_event;
        return simple_parallel_for<KernelName>(range(num_work_items), std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<1> num_work_items, event dep_event, Rest &&...rest) {
        (void)dep_event;
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<2> num_work_items, event dep_event, Rest &&...rest) {
        (void)dep_event;
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<3> num_work_items, event dep_event, Rest &&...rest) {
        (void)dep_event;
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(size_t num_work_items, const std::vector<event> &dep_events, Rest &&...rest) {
        (void)dep_events;
        return simple_parallel_for<KernelName>(range(num_work_items), std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<1> num_work_items, const std::vector<event> &dep_events, Rest &&...rest) {
        (void)dep_events;
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<2> num_work_items, const std::vector<event> &dep_events, Rest &&...rest) {
        (void)dep_events;
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(range<3> num_work_items, const std::vector<event> &dep_events, Rest &&...rest) {
        (void)dep_events;
        return simple_parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dims, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(nd_range<Dims> execution_range, Rest &&...rest) {
        return parallel_for_nd_range<KernelName>(execution_range, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dims, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(nd_range<Dims> execution_range, event dep_event, Rest &&...rest) {
        (void)dep_event;
        return parallel_for_nd_range<KernelName>(execution_range, std::forward<Rest>(rest)...);
    }

    template<typename KernelName = simsycl::detail::unnamed_kernel, int Dims, typename... Rest,
        std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for(nd_range<Dims> execution_range, const std::vector<event> &dep_events, Rest &&...rest) {
        (void)dep_events;
        return parallel_for_nd_range<KernelName>(execution_range, std::forward<Rest>(rest)...);
    }

    /* -- USM functions -- */

    event memcpy(void *dest, const void *src, size_t num_bytes) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        ::memcpy(dest, src, num_bytes);
        return status.end();
    }

    event memcpy(void *dest, const void *src, size_t num_bytes, event /* dep_event */) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        ::memcpy(dest, src, num_bytes);
        return status.end();
    }

    event memcpy(void *dest, const void *src, size_t num_bytes, const std::vector<event> & /* dep_events */) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        ::memcpy(dest, src, num_bytes);
        return status.end();
    }

    template<typename T>
    event copy(const T *src, T *dest, size_t count) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        std::copy_n(src, count, dest);
        return status.end();
    }

    template<typename T>
    event copy(const T *src, T *dest, size_t count, event dep_event) {
        (void)(dep_event);
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        std::copy_n(src, count, dest);
        return status.end();
    }

    template<typename T>
    event copy(const T *src, T *dest, size_t count, const std::vector<event> &dep_events) {
        (void)(dep_events);
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        std::copy_n(src, count, dest);
        return status.end();
    }

    event memset(void *ptr, int value, size_t num_bytes) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        ::memset(ptr, value, num_bytes);
        return status.end();
    }

    event memset(void *ptr, int value, size_t num_bytes, event /* dep_event */) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        ::memset(ptr, value, num_bytes);
        return status.end();
    }

    event memset(void *ptr, int value, size_t num_bytes, const std::vector<event> & /* dep_events */) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        ::memset(ptr, value, num_bytes);
        return status.end();
    }

    template<typename T>
    event fill(void *ptr, const T &pattern, size_t count) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        std::fill_n(static_cast<T *>(ptr), count, pattern);
        return status.end();
    }

    template<typename T>
    event fill(void *ptr, const T &pattern, size_t count, event /* dep_event */) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        std::fill_n(static_cast<T *>(ptr), count, pattern);
        return status.end();
    }

    template<typename T>
    event fill(void *ptr, const T &pattern, size_t count, const std::vector<event> & /* dep_events */) {
        auto status = detail::event_state::submit();
        detail::system_lock lock; // implicitly enforce dependency ordering by keeping a lock within every task
        status.start();
        std::fill_n(static_cast<T *>(ptr), count, pattern);
        return status.end();
    }

    event prefetch(void * /* ptr */, size_t /* num_bytes */) { return detail::event_state::instant(); }

    event prefetch(void * /* ptr */, size_t /* num_bytes */, event /* dep_event */) {
        return detail::event_state::instant();
    }

    event prefetch(void * /* ptr */, size_t /* num_bytes */, const std::vector<event> & /* dep_events */) {
        return detail::event_state::instant();
    }

    event mem_advise(void * /* ptr */, size_t /* num_bytes */, int /* advice */) {
        return detail::event_state::instant();
    }

    event mem_advise(void * /* ptr */, size_t /* num_bytes */, int /* advice */, event /* dep_event */) {
        return detail::event_state::instant();
    }

    event mem_advise(
        void * /* ptr */, size_t /* num_bytes */, int /* advice */, const std::vector<event> & /* dep_events */) {
        return detail::event_state::instant();
    }

    /// Placeholder accessor shortcuts

    // Explicit copy functions

    // access::placeholder
    SIMSYCL_START_IGNORING_DEPRECATIONS

    template<typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src, std::shared_ptr<DestT> dest) {
        return submit([=](handler &cgh) {
            cgh.require(src);
            cgh.copy(src, dest);
        });
    }

    template<typename SrcT, typename DestT, int DestDims, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    event copy(std::shared_ptr<SrcT> src, accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest) {
        return submit([=](handler &cgh) {
            cgh.require(dest);
            cgh.copy(src, dest);
        });
    }

    template<typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt, access::placeholder IsPlaceholder,
        typename DestT>
    event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src, DestT *dest) {
        return submit([=](handler &cgh) {
            cgh.require(src);
            cgh.copy(src, dest);
        });
    }

    template<typename SrcT, typename DestT, int DestDims, access_mode DestMode, target DestTgt,
        access::placeholder IsPlaceholder>
    event copy(const SrcT *src, accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest) {
        return submit([=](handler &cgh) {
            cgh.require(dest);
            cgh.copy(src, dest);
        });
    }

    template<typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt, access::placeholder IsSrcPlaceholder,
        typename DestT, int DestDims, access_mode DestMode, target DestTgt, access::placeholder IsDestPlaceholder>
    event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsSrcPlaceholder> src,
        accessor<DestT, DestDims, DestMode, DestTgt, IsDestPlaceholder> dest) {
        return submit([=](handler &cgh) {
            cgh.require(src);
            cgh.require(dest);
            cgh.copy(src, dest);
        });
    }

    template<typename T, int Dims, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    event update_host(accessor<T, Dims, Mode, Tgt, IsPlaceholder> acc) {
        return submit([=](handler &cgh) {
            cgh.require(acc);
            cgh.update_host(acc);
        });
    }

    template<typename T, int Dims, access_mode Mode, target Tgt, access::placeholder IsPlaceholder>
    event fill(accessor<T, Dims, Mode, Tgt, IsPlaceholder> dest, const T &src) {
        return submit([=](handler &cgh) {
            cgh.require(dest);
            cgh.fill(dest, src);
        });
    }

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

  private:
    template<typename>
    friend class detail::weak_ref;

    struct internal_t {
    } inline static constexpr internal{};

    queue(std::shared_ptr<detail::queue_state> &&state) : reference_type(std::move(state)) {}

    explicit queue(internal_t /* tag */, const detail::device_selector &selector, const async_handler &async_handler,
        const property_list &prop_list);

    explicit queue(internal_t /* tag */, const device &sycl_device, const async_handler &async_handler,
        const property_list &prop_list);

    explicit queue(internal_t /* tag */, const context &sycl_context, const detail::device_selector &selector,
        const async_handler &async_handler, const property_list &prop_list);

    explicit queue(internal_t /* tag */, const context &sycl_context, const device &sycl_device,
        const async_handler &async_handler, const property_list &prop_list);

    template<typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event simple_parallel_for(range<Dims> num_work_items, Rest &&...rest) {
        return submit([&](handler &cgh) { cgh.parallel_for<KernelName>(num_work_items, std::forward<Rest>(rest)...); });
    }

    template<typename KernelName, int Dims, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    event parallel_for_nd_range(nd_range<Dims> execution_range, Rest &&...rest) {
        return submit(
            [&](handler &cgh) { cgh.parallel_for<KernelName>(execution_range, std::forward<Rest>(rest)...); });
    }
};

} // namespace simsycl::sycl

template<>
struct std::hash<simsycl::sycl::queue>
    : std::hash<simsycl::detail::reference_type<simsycl::sycl::queue, simsycl::detail::queue_state>> {};
