#pragma once

#include "item.hh"

namespace simsycl::detail {

template<int Dimensions>
sycl::h_item<Dimensions> make_h_item(const sycl::item<Dimensions, false> &global_item,
    const sycl::item<Dimensions, false> &logical_local_item, const sycl::item<Dimensions, false> &physical_local_item) {
    return sycl::h_item<Dimensions>(global_item, logical_local_item, physical_local_item);
}

} // namespace simsycl::detail

namespace simsycl::sycl {

template<int Dimensions>
class h_item {
  public:
    static constexpr int dimensions = Dimensions;

    h_item() = delete;

    item<Dimensions, false> get_global() const { return m_global_item; }

    item<Dimensions, false> get_local() const { return m_logical_local_item; }

    item<Dimensions, false> get_logical_local() const { return m_logical_local_item; }

    item<Dimensions, false> get_physical_local() const { return m_physical_local_item; }

    range<Dimensions> get_global_range() const { return m_global_item.get_range(); }

    size_t get_global_range(int dimension) const { return m_global_item.get_range(dimension); }

    id<Dimensions> get_global_id() const { return m_global_item.get_id(); }

    size_t get_global_id(int dimension) const { return m_global_item.get_id(dimension); }

    range<Dimensions> get_local_range() const { return m_logical_local_item.get_range(); }

    size_t get_local_range(int dimension) const { return m_logical_local_item.get_range(dimension); }

    id<Dimensions> get_local_id() const { return m_logical_local_item.get_id(); }

    size_t get_local_id(int dimension) const { return m_logical_local_item.get_id(dimension); }

    range<Dimensions> get_logical_local_range() const { return m_logical_local_item.get_range(); }

    size_t get_logical_local_range(int dimension) const { return m_logical_local_item.get_range(dimension); }

    id<Dimensions> get_logical_local_id() const { return m_logical_local_item.get_id(); }

    size_t get_logical_local_id(int dimension) const { return m_logical_local_item.get_id(dimension); }

    range<Dimensions> get_physical_local_range() const { return m_physical_local_item.get_range(); }

    size_t get_physical_local_range(int dimension) const { return m_physical_local_item.get_range(dimension); }

    id<Dimensions> get_physical_local_id() const { return m_physical_local_item.get_id(); }

    size_t get_physical_local_id(int dimension) const { return m_physical_local_item.get_id(dimension); }

    friend bool operator==(const h_item &lhs, const h_item &rhs) {
        return lhs.m_global_item == rhs.m_global_item && lhs.m_logical_local_item == rhs.m_logical_local_item
            && lhs.m_physical_local_item == rhs.m_physical_local_item;
    }

    friend bool operator!=(const h_item &lhs, const h_item &rhs) { return !(lhs == rhs); }

  private:
    item<Dimensions, false> m_global_item;
    item<Dimensions, false> m_logical_local_item;
    item<Dimensions, false> m_physical_local_item;

    friend sycl::h_item<Dimensions> simsycl::detail::make_h_item<Dimensions>(const sycl::item<Dimensions, false> &,
        const sycl::item<Dimensions, false> &, const sycl::item<Dimensions, false> &);

    h_item(const sycl::item<Dimensions, false> &global_item, const sycl::item<Dimensions, false> &logical_local_item,
        const sycl::item<Dimensions, false> &physical_local_item)
        : m_global_item(global_item), m_logical_local_item(logical_local_item),
          m_physical_local_item(physical_local_item) {}
};

} // namespace simsycl::sycl