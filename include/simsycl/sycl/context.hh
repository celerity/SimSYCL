#pragma once

#include "async_handler.hh"
#include "forward.hh"
#include "property.hh"

#include "../detail/reference_type.hh"

#include <vector>


namespace simsycl::detail {

struct context_state;

} // namespace simsycl::detail

namespace simsycl::sycl {

class context final : public detail::reference_type<context, detail::context_state>, public detail::property_interface {
  private:
    using reference_type = detail::reference_type<context, detail::context_state>;
    using property_compatibility
        = detail::property_compatibility_with<context /* apparently no compatible properties? */>;

  public:
    explicit context(const property_list &prop_list = {});

    explicit context(async_handler async_handler, const property_list &prop_list = {});

    explicit context(const device &dev, const property_list &prop_list = {});

    explicit context(const device &dev, async_handler async_handler, const property_list &prop_list = {});

    explicit context(const std::vector<device> &device_list, const property_list &prop_list = {});

    explicit context(
        const std::vector<device> &device_list, async_handler async_handler, const property_list &prop_list = {});

    backend get_backend() const noexcept { return backend::simsycl; }

    platform get_platform() const;

    std::vector<device> get_devices() const;

    template<typename Param>
    typename Param::return_type get_info() const;

    template<typename Param>
    typename Param::return_type get_backend_info() const;

  private:
    template<typename>
    friend class detail::weak_ref;

    struct internal_t {
    } inline static constexpr internal{};

    explicit context(internal_t, const std::vector<device> &devices, const async_handler &async_handler,
        const property_list &prop_list);
    context(std::shared_ptr<detail::context_state> &&state) : reference_type(std::move(state)) {}
};

} // namespace simsycl::sycl

template<>
struct std::hash<simsycl::sycl::context>
    : public std::hash<simsycl::detail::reference_type<simsycl::sycl::context, simsycl::detail::context_state>> {};
