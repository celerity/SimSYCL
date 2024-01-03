#pragma once

#include "group.hh"
#include "h_item.hh"

#include <vector>

namespace simsycl::sycl {

template<typename T, int Dimensions = 1>
class private_memory {
  public:
    private_memory(const group<Dimensions> &group) : m_group(group) {}

    // Access the instance for the current work-item
    // Construct the storage if it has not yet been constructed
    T &operator()(const h_item<Dimensions> &id) {
        if(m_data.empty()) {
            size_t num_items = m_group.get_local_linear_range();
            m_data.resize(num_items);
        }
        return m_data[id.get_local().get_linear_id()];
    }

  private:
    std::vector<T> m_data;
    const group<Dimensions> &m_group;
};

} // namespace simsycl::sycl
