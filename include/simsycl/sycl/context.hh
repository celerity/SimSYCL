#pragma once

#include "exception.hh"
#include "forward.hh"
#include "property.hh"

namespace simsycl::sycl {

class context: public detail::property_interface<context /* apparently no compatible properties? */> {
 public:
  explicit context(const property_list& prop_list = {});

  explicit context(async_handler async_handler,
                   const property_list& prop_list = {});

  explicit context(const device& dev, const property_list& prop_list = {});

  explicit context(const device& dev, async_handler async_handler,
                   const property_list& prop_list = {});

  explicit context(const std::vector<device>& device_list,
                   const property_list& prop_list = {});

  explicit context(const std::vector<device>& device_list,
                   async_handler async_handler,
                   const property_list& prop_list = {});

  /* -- common interface members -- */

  backend get_backend() const noexcept;

  platform get_platform() const;

  std::vector<device> get_devices() const;

  template <typename Param> typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;
};

} // namespace sycl
