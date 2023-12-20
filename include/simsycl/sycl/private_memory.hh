#pragma once

#include "group.hh"
#include "h_item.hh"

#include <vector>

namespace simsycl::sycl {

template<typename T, int Dimensions = 1>
class private_memory {
  public:
    // Construct based directly off the number of work-items
    private_memory(const group<Dimensions> &group) { m_data.resize(group.get_local_linear_range()); }

    // Access the instance for the current work-item
    T &operator()(const h_item<Dimensions> &id) { return m_data[id.get_local_linear_id()]; }

  private:
    std::vector<T> m_data;
};

} // namespace simsycl::sycl
