#pragma once

#include "group.hh"
#include "h_item.hh"

#include <vector>

namespace simsycl::sycl {

template<typename T, int Dimensions = 1>
class private_memory {
  public:
    private_memory(const group<Dimensions> &group) : m_data(group.get_local_linear_range()) {}

    // Access the instance for the current work-item by physical id
    // Construct the storage if it has not yet been constructed
    T &operator()(const h_item<Dimensions> &id) {
        SIMSYCL_CHECK(id.get_physical_local().get_linear_id() < m_data.size());
        return m_data[id.get_physical_local().get_linear_id()];
    }

  private:
    std::vector<T> m_data;
};

} // namespace simsycl::sycl
