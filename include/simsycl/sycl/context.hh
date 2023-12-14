#pragma once

#include "async_handler.hh"
#include "forward.hh"
#include "property.hh"

#include "../detail/reference_type.hh"


namespace simsycl::detail {

struct context_state;

} // namespace simsycl::detail

namespace simsycl::sycl {

class context final : public detail::reference_type<context, detail::context_state>, public detail::property_interface {
  private:
    using reference_type = detail::reference_type<context, detail::context_state>;
    using property_compatibilty
        = detail::property_compatibility_with<context /* apparently no compatible properties? */>;

  public:
    explicit context(const property_list &prop_list = {});

    explicit context(async_handler async_handler, const property_list &prop_list = {});

    explicit context(const device &dev, const property_list &prop_list = {});

    explicit context(const device &dev, async_handler async_handler, const property_list &prop_list = {});

    explicit context(const std::vector<device> &device_list, const property_list &prop_list = {});

    explicit context(
        const std::vector<device> &device_list, async_handler async_handler, const property_list &prop_list = {});

    backend get_backend() const noexcept;

    platform get_platform() const;

    std::vector<device> get_devices() const;

    template<typename Param>
    typename Param::return_type get_info() const {
        return {};
    }

    template<typename Param>
    typename Param::return_type get_backend_info() const {
        return {};
    }
};

} // namespace simsycl::sycl
