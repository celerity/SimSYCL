#pragma once

#include "enums.hh"
#include "exception.hh"
#include "forward.hh"

namespace simsycl::sycl {

// SimSYCL does not expose any backends, but the function prototypes are made available for generic code

template <backend Backend>
class backend_traits {
  public:
    template <class T>
    using input_type = void;

    template <class T>
    using return_type = void;

    using errc = void;
};

template <backend Backend, typename SyclType>
using backend_input_t = typename backend_traits<Backend>::template input_type<SyclType>;

template <backend Backend, typename SyclType>
using backend_return_t = typename backend_traits<Backend>::template return_type<SyclType>;

template <backend Backend, class T>
backend_return_t<Backend, T> get_native(const T &sycl_object);

template <backend Backend>
platform make_platform(const backend_input_t<Backend, platform> &backend_object);

template <backend Backend>
device make_device(const backend_input_t<Backend, device> &backend_object);

template <backend Backend>
context make_context(const backend_input_t<Backend, context> &backend_object, const async_handler async_handler = {});

template <backend Backend>
queue make_queue(const backend_input_t<Backend, queue> &backend_object, const context &target_context,
    const async_handler async_handler = {});

template <backend Backend>
event make_event(const backend_input_t<Backend, event> &backend_object, const context &target_context);

template <backend Backend, typename T, int Dimensions = 1,
    typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
buffer<T, Dimensions, AllocatorT> make_buffer(
    const backend_input_t<Backend, buffer<T, Dimensions, AllocatorT>> &backend_object, const context &target_context,
    event available_event);

template <backend Backend, typename T, int Dimensions = 1,
    typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
buffer<T, Dimensions, AllocatorT> make_buffer(
    const backend_input_t<Backend, buffer<T, Dimensions, AllocatorT>> &backend_object, const context &target_context);

template <backend Backend, int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
sampled_image<Dimensions, AllocatorT> make_sampled_image(
    const backend_input_t<Backend, sampled_image<Dimensions, AllocatorT>> &backend_object,
    const context &target_context, image_sampler image_sampler, event available_event);

template <backend Backend, int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
sampled_image<Dimensions, AllocatorT> make_sampled_image(
    const backend_input_t<Backend, sampled_image<Dimensions, AllocatorT>> &backend_object,
    const context &target_context, image_sampler image_sampler);

template <backend Backend, int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
unsampled_image<Dimensions, AllocatorT> make_unsampled_image(
    const backend_input_t<Backend, unsampled_image<Dimensions, AllocatorT>> &backend_object,
    const context &target_context, event available_event);

template <backend Backend, int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
unsampled_image<Dimensions, AllocatorT> make_unsampled_image(
    const backend_input_t<Backend, unsampled_image<Dimensions, AllocatorT>> &backend_object,
    const context &target_context);

template <backend Backend, bundle_state State>
kernel_bundle<State> make_kernel_bundle(
    const backend_input_t<Backend, kernel_bundle<State>> &backend_object, const context &target_context);

template <backend Backend>
kernel make_kernel(const backend_input_t<Backend, kernel> &backend_object, const context &target_context);

} // namespace simsycl::sycl
