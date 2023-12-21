#pragma once

#include "backend.hh"
#include "forward.hh"


namespace simsycl::sycl {

class interop_handle {
  public:
    backend get_backend() const noexcept { return backend::simsycl; }

    SIMSYCL_START_IGNORING_DEPRECATIONS

    template<backend Backend, typename DataT, int Dims, access_mode AccMode, target AccTarget,
        access::placeholder IsPlaceholder>
    backend_return_t<Backend, buffer<DataT, Dims>> get_native_mem(const accessor<DataT, Dims, AccMode, AccTarget, // (1)
        IsPlaceholder> &buffer_acc) const;

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

    template<backend Backend, typename DataT, int Dims, access_mode AccMode>
    backend_return_t<Backend, unsampled_image<Dims>> get_native_mem(
        const unsampled_image_accessor<DataT, Dims, AccMode, image_target::device> &image_acc) const;

    template<backend Backend, typename DataT, int Dims>
    backend_return_t<Backend, sampled_image<Dims>> get_native_mem(
        const sampled_image_accessor<DataT, Dims, image_target::device> &image_acc) const;

    template<backend Backend>
    backend_return_t<Backend, queue> get_native_queue() const;

    template<backend Backend>
    backend_return_t<Backend, device> get_native_device() const;

    template<backend Backend>
    backend_return_t<Backend, context> get_native_context() const;

  private:
    friend interop_handle detail::make_interop_handle();

    interop_handle() = default;
};

} // namespace simsycl::sycl

namespace simsycl::detail {

inline sycl::interop_handle make_interop_handle() { return {}; }

} // namespace simsycl::detail
