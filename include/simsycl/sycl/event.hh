#pragma once

#include "forward.hh"

#include <vector>

namespace simsycl::sycl {

class event {
 public:
  event() = default;

  /* -- common interface members -- */

  backend get_backend() const noexcept;

  std::vector<event> get_wait_list();

  void wait() {}

  static void wait(const std::vector<event>& /* event_list */) {}

  void wait_and_throw() {}

  static void wait_and_throw(const std::vector<event>& /* event_list */) {}

  template <typename Param> typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;

  template <typename Param>
  typename Param::return_type get_profiling_info() const;
};

} // namespace sycl
