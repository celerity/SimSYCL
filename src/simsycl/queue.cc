#include "simsycl/sycl/queue.hh"
#include "simsycl/sycl/context.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/info.hh"

#include "simsycl/system.hh"


namespace simsycl::detail {

struct queue_state {
    sycl::device device;
    sycl::context context;
    sycl::async_handler async_handler;

    queue_state(const device_selector &selector, const sycl::async_handler &async_handler)
        : device(select_device(selector)), context(device, async_handler), async_handler(async_handler) {}

    queue_state(const sycl::device &device, const sycl::async_handler &async_handler)
        : device(device), context(device, async_handler), async_handler(async_handler) {}

    queue_state(const sycl::device &device, const sycl::context &context, const sycl::async_handler &async_handler)
        : device(device), context(context), async_handler(async_handler) {}
};

} // namespace simsycl::detail

namespace simsycl::sycl {

queue::queue(internal_t /* tag */, const detail::device_selector &selector, const async_handler &async_handler,
    const property_list &prop_list)
    : reference_type(std::in_place, selector, async_handler), property_interface(prop_list, property_compatibility()) {}

queue::queue(
    internal_t /* tag */, const device &sycl_device, const async_handler &async_handler, const property_list &prop_list)
    : reference_type(std::in_place, sycl_device, async_handler),
      property_interface(prop_list, property_compatibility()) {}

queue::queue(internal_t /* tag */, const context &sycl_context, const device &sycl_device,
    const async_handler &async_handler, const property_list &prop_list)
    : reference_type(std::in_place, sycl_device, sycl_context, async_handler),
      property_interface(prop_list, property_compatibility()) {}

queue::queue(const property_list &prop_list) : queue(internal, default_selector_v, async_handler{}, prop_list) {}

queue::queue(const async_handler &async_handler, const property_list &prop_list)
    : queue(internal, default_selector_v, async_handler, prop_list) {}

queue::queue(const device &sycl_device, const property_list &prop_list)
    : queue(internal, sycl_device, async_handler{}, prop_list) {}

queue::queue(const device &sycl_device, const async_handler &async_handler, const property_list &prop_list)
    : queue(internal, sycl_device, async_handler, prop_list) {}

template<DeviceSelector Selector>
queue::queue(const context &sycl_context, const Selector &device_selector, const property_list &prop_list)
    : queue(internal, device_selector, sycl_context, async_handler{}, prop_list) {}

template<DeviceSelector Selector>
queue::queue(const context &sycl_context, const Selector &device_selector, const async_handler &async_handler,
    const property_list &prop_list)
    : queue(internal, device_selector, sycl_context, async_handler, prop_list) {}

queue::queue(const context &sycl_context, const device &sycl_device, const property_list &prop_list)
    : queue(internal, sycl_context, sycl_device, async_handler{}, prop_list) {}

queue::queue(const context &sycl_context, const device &sycl_device, const async_handler &async_handler,
    const property_list &prop_list)
    : queue(internal, sycl_context, sycl_device, async_handler, prop_list) {}

template<>
context queue::get_info<info::queue::context>() const {
    return state().context;
}

template<>
device queue::get_info<info::queue::device>() const {
    return state().device;
}

backend queue::get_backend() const noexcept { return backend::simsycl; }

context queue::get_context() const { return state().context; }

device queue::get_device() const { return state().device; }

} // namespace simsycl::sycl
