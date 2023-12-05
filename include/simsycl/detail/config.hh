#pragma once

#include <inttypes.h>

namespace simsycl::detail {

class config {
  public:
    inline static uint32_t max_sub_group_size = 32;
};

template <typename T>
class configure_temporarily {
  public:
    configure_temporarily(T &to_configure, T new_value) : m_to_configure(to_configure) {
        m_old_value = to_configure;
        to_configure = new_value;
    }
    ~configure_temporarily() { m_to_configure = m_old_value; }

  private:
    T &m_to_configure;
    T m_old_value;
};

} // namespace simsycl::detail
