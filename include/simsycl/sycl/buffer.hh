#pragma once

#include "forward.hh"
#include "property.hh"

#include <memory>
#include <mutex>
#include <type_traits>
#include <vector> // for std::data

namespace simsycl::sycl::property::buffer {

class use_host_ptr {
  public:
    use_host_ptr() = default;
};

class use_mutex {
  public:
    use_mutex(std::mutex &mutex_ref);

    std::mutex *get_mutex_ptr() const;
};

class context_bound {
  public:
    context_bound(context bound_context);

    context get_context() const;
};

} // namespace simsycl::sycl::property::buffer

namespace simsycl::sycl {

template <>
struct is_property<property::buffer::use_host_ptr> : public std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_host_ptr, buffer<T, Dimensions, AllocatorT>> : public std::true_type {};

template <>
struct is_property<property::buffer::use_mutex> : public std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_mutex, buffer<T, Dimensions, AllocatorT>> : public std::true_type {};

template <>
struct is_property<property::buffer::context_bound> : public std::true_type {};

template <typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::context_bound, buffer<T, Dimensions, AllocatorT>> : public std::true_type {};

} // namespace simsycl::sycl

namespace simsycl::sycl {

template <typename T, int Dimensions, typename AllocatorT>
class buffer : public detail::property_interface<buffer<T>, property::buffer::use_host_ptr, property::buffer::use_mutex,
                   property::buffer::context_bound> {
  public:
    using value_type = T;
    using reference = value_type &;
    using const_reference = const value_type &;
    using allocator_type = AllocatorT;

    buffer(const range<Dimensions> &buffer_range, const property_list &prop_list = {});

    buffer(const range<Dimensions> &buffer_range, AllocatorT allocator, const property_list &prop_list = {});

    buffer(T *host_data, const range<Dimensions> &buffer_range, const property_list &prop_list = {});

    buffer(
        T *host_data, const range<Dimensions> &buffer_range, AllocatorT allocator, const property_list &prop_list = {});

    buffer(const T *host_data, const range<Dimensions> &buffer_range, const property_list &prop_list = {});

    buffer(const T *host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {});

    template <typename Container,
        std::enable_if_t<std::is_convertible_v<decltype(std::data(std::declval<Container &>())), T *>
                && std::is_convertible_v<decltype(std::size(std::declval<Container &>())), size_t> && Dimensions == 1,
            int>
        = 0>
    buffer(Container &container, AllocatorT allocator, const property_list &prop_list = {});

    template <typename Container,
        std::enable_if_t<std::is_convertible_v<decltype(std::data(std::declval<Container &>())), T *>
                && std::is_convertible_v<decltype(std::size(std::declval<Container &>())), size_t> && Dimensions == 1,
            int>
        = 0>
    buffer(Container &container, const property_list &prop_list = {});

    buffer(const std::shared_ptr<T> &host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {});

    buffer(const std::shared_ptr<T> &host_data, const range<Dimensions> &buffer_range,
        const property_list &prop_list = {});

    buffer(const std::shared_ptr<T[]> &host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {});

    buffer(const std::shared_ptr<T[]> &host_data, const range<Dimensions> &buffer_range,
        const property_list &prop_list = {});

    template <class InputIterator, std::enable_if_t<Dimensions == 1, int> = 0>
    buffer(InputIterator first, InputIterator last, AllocatorT allocator, const property_list &prop_list = {});

    template <class InputIterator, std::enable_if_t<Dimensions == 1, int> = 0>
    buffer(InputIterator first, InputIterator last, const property_list &prop_list = {});

    buffer(buffer &b, const id<Dimensions> &base_index, const range<Dimensions> &sub_range);

    /* -- common interface members -- */

    range<Dimensions> get_range() const;

    size_t byte_size() const noexcept;

    size_t size() const noexcept;

    // Deprecated
    size_t get_count() const;

    // Deprecated
    size_t get_size() const;

    AllocatorT get_allocator() const;

    template <access_mode Mode = access_mode::read_write, target Targ = target::device>
    accessor<T, Dimensions, Mode, Targ> get_access(handler &command_group_handler);

    // Deprecated
    template <access_mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer> get_access();

    template <access_mode Mode = access_mode::read_write, target Targ = target::device>
    accessor<T, Dimensions, Mode, Targ> get_access(
        handler &command_group_handler, range<Dimensions> access_range, id<Dimensions> access_offset = {});

    // Deprecated
    template <access_mode Mode>
    accessor<T, Dimensions, Mode, target::host_buffer> get_access(
        range<Dimensions> access_range, id<Dimensions> access_offset = {});

    template <typename... Ts>
    auto get_access(Ts...);

    template <typename... Ts>
    auto get_host_access(Ts...);

    template <typename Destination = std::nullptr_t>
    void set_final_data(Destination final_data = nullptr);

    void set_write_back(bool flag = true);

    bool is_sub_buffer() const;

    template <typename ReinterpretT, int ReinterpretDim>
    buffer<ReinterpretT, ReinterpretDim,
        typename std::allocator_traits<AllocatorT>::template rebind_alloc<ReinterpretT>>
    reinterpret(range<ReinterpretDim> reinterpret_range) const;

    template <typename ReinterpretT, int ReinterpretDim = Dimensions,
        std::enable_if_t<ReinterpretDim == 1 || (ReinterpretDim == Dimensions && sizeof(ReinterpretT) == sizeof(T)),
            int>
        = 0>
    buffer<ReinterpretT, ReinterpretDim,
        typename std::allocator_traits<AllocatorT>::template rebind_alloc<ReinterpretT>>
    reinterpret() const;
};

// Deduction guides
template <class InputIterator, class AllocatorT>
buffer(InputIterator, InputIterator, AllocatorT, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1, AllocatorT>;

template <class InputIterator>
buffer(InputIterator, InputIterator, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;

template <class T, int Dimensions, class AllocatorT>
buffer(const T *, const range<Dimensions> &, AllocatorT, const property_list & = {})
    -> buffer<T, Dimensions, AllocatorT>;

template <class T, int Dimensions>
buffer(const T *, const range<Dimensions> &, const property_list & = {}) -> buffer<T, Dimensions>;

template <class Container, class AllocatorT>
buffer(Container &, AllocatorT, const property_list & = {}) -> buffer<typename Container::value_type, 1, AllocatorT>;

template <class Container>
buffer(Container &, const property_list & = {}) -> buffer<typename Container::value_type, 1>;

} // namespace simsycl::sycl
