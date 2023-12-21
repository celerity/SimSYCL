#pragma once

#include "context.hh"
#include "forward.hh"
#include "property.hh"

#include "../detail/allocation.hh"
#include "../detail/reference_type.hh"

#include <concepts>
#include <cstring>
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
    use_mutex(std::mutex &mutex_ref) : m_mutex(&mutex_ref) {}

    std::mutex *get_mutex_ptr() const { return m_mutex; }

  private:
    std::mutex *m_mutex;
};

class context_bound {
  public:
    context_bound(context bound_context) : m_context(std::move(bound_context)) {}

    context get_context() const;

  private:
    context m_context;
};

} // namespace simsycl::sycl::property::buffer

namespace simsycl::sycl {

template<>
struct is_property<property::buffer::use_host_ptr> : std::true_type {};

template<typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_host_ptr, buffer<T, Dimensions, AllocatorT>> : std::true_type {};

template<>
struct is_property<property::buffer::use_mutex> : std::true_type {};

template<typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_mutex, buffer<T, Dimensions, AllocatorT>> : std::true_type {};

template<>
struct is_property<property::buffer::context_bound> : std::true_type {};

template<typename T, int Dimensions, typename AllocatorT>
struct is_property_of<property::buffer::context_bound, buffer<T, Dimensions, AllocatorT>> : std::true_type {};

} // namespace simsycl::sycl

namespace simsycl::detail {

template<typename C, typename T>
constexpr bool is_container_v = requires(C &c) {
    { std::data(c) } -> std::convertible_to<T *>;
    { std::size(c) } -> std::convertible_to<size_t>;
};

template<typename C, typename T>
concept Container = is_container_v<C, T>;

template<typename T, int Dimensions, typename AllocatorT>
struct buffer_state {
    sycl::range<Dimensions> range;
    AllocatorT allocator;
    T *buffer;
    T *write_back_to = nullptr; // TODO has to be an arbitrary output iterator (aka a type-erased lambda)

    buffer_state(sycl::range<Dimensions> range, const AllocatorT &allocator = {}, T *write_back_to = nullptr)
        : range(range), allocator(allocator), buffer(this->allocator.allocate(range.size())),
          write_back_to(write_back_to) {
        memset(buffer, static_cast<int>(detail::uninitialized_memory_pattern), range.size() * sizeof(T));
    }

    buffer_state(const buffer_state &) = delete;
    buffer_state(buffer_state &&) = delete;
    buffer_state &operator=(const buffer_state &) = delete;
    buffer_state &operator=(buffer_state &&) = delete;

    ~buffer_state() {
        if(write_back_to != nullptr) { std::copy_n(buffer, range.size(), write_back_to); }
        allocator.deallocate(buffer, range.size());
    }
};

} // namespace simsycl::detail

namespace simsycl::sycl {

template<typename T, int Dimensions, typename AllocatorT>
class buffer final : public detail::reference_type<buffer<T, Dimensions, AllocatorT>,
                         detail::buffer_state<std::remove_const_t<T>, Dimensions, AllocatorT>>,
                     public detail::property_interface {
  private:
    using reference_type
        = detail::reference_type<buffer<T, Dimensions, AllocatorT>, detail::buffer_state<T, Dimensions, AllocatorT>>;
    using property_compatibility = detail::property_compatibility_with<buffer<T, Dimensions, AllocatorT>,
        property::buffer::use_host_ptr, property::buffer::use_mutex, property::buffer::context_bound>;
    using typename reference_type::state_type;

  public:
    using value_type = T;
    using reference = value_type &;
    using const_reference = const value_type &;
    using allocator_type = AllocatorT;

    buffer(const range<Dimensions> &buffer_range, const property_list &prop_list = {})
        : buffer(buffer_range, AllocatorT(), prop_list) {}

    buffer(const range<Dimensions> &buffer_range, AllocatorT allocator, const property_list &prop_list = {})
        : reference_type(std::in_place, buffer_range, allocator),
          property_interface(prop_list, property_compatibility()) {}

    buffer(T *host_data, const range<Dimensions> &buffer_range, const property_list &prop_list = {})
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    buffer(
        T *host_data, const range<Dimensions> &buffer_range, AllocatorT allocator, const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          reference_type(std::in_place, buffer_range, allocator, host_data) {
        std::copy_n(host_data, state().range.size(), state().buffer);
    }

    buffer(const T *host_data, const range<Dimensions> &buffer_range, const property_list &prop_list = {})
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    buffer(const T *host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          reference_type(std::in_place, buffer_range, allocator) {
        std::copy_n(host_data, state().range.size(), state().buffer);
    }

    template<simsycl::detail::Container<T> Container>
        requires(Dimensions == 1)
    buffer(Container &container, AllocatorT allocator, const property_list &prop_list = {});

    template<simsycl::detail::Container<T> Container>
        requires(Dimensions == 1)
    buffer(Container &container, const property_list &prop_list = {}) : buffer(container, AllocatorT(), prop_list) {}

    buffer(const std::shared_ptr<T> &host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {});

    buffer(
        const std::shared_ptr<T> &host_data, const range<Dimensions> &buffer_range, const property_list &prop_list = {})
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    buffer(const std::shared_ptr<T[]> &host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {});

    buffer(const std::shared_ptr<T[]> &host_data, const range<Dimensions> &buffer_range,
        const property_list &prop_list = {})
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    template<class InputIterator>
        requires(Dimensions == 1)
    buffer(InputIterator first, InputIterator last, AllocatorT allocator, const property_list &prop_list = {});

    template<class InputIterator>
        requires(Dimensions == 1)
    buffer(InputIterator first, InputIterator last, const property_list &prop_list = {})
        : buffer(first, last, AllocatorT(), prop_list) {}

    buffer(buffer &b, const id<Dimensions> &base_index, const range<Dimensions> &sub_range);

    range<Dimensions> get_range() const { return state().range; }

    size_t byte_size() const noexcept { return size() * sizeof(T); }

    size_t size() const noexcept { return get_range().size(); }

    // Deprecated
    size_t get_count() const { return size(); }

    // Deprecated
    size_t get_size() const { return byte_size(); }

    AllocatorT get_allocator() const { return state().allocator; }

    template<access_mode Mode = access_mode::read_write, target Targ = target::device>
    accessor<T, Dimensions, Mode, Targ> get_access(handler &command_group_handler) {
        return accessor<T, Dimensions, Mode, Targ>(*this, command_group_handler);
    }

    template<access_mode Mode = access_mode::read_write, target Targ = target::device>
    accessor<T, Dimensions, Mode, Targ> get_access(
        handler &command_group_handler, range<Dimensions> access_range, id<Dimensions> access_offset = {}) {
        return accessor<T, Dimensions, Mode, Targ>(*this, command_group_handler, access_range, access_offset);
    }

    SIMSYCL_START_IGNORING_DEPRECATIONS

    template<access_mode Mode>
    [[deprecated]] accessor<T, Dimensions, Mode, target::host_buffer> get_access() {
        accessor<T, Dimensions, Mode, target::host_buffer>(*this);
    }

    template<access_mode Mode>
    [[deprecated]] accessor<T, Dimensions, Mode, target::host_buffer> get_access(
        range<Dimensions> access_range, id<Dimensions> access_offset = {}) {
        accessor<T, Dimensions, Mode, target::host_buffer>(*this, access_range, access_offset);
    }

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

    template<typename... Ts>
    auto get_access(Ts... args) {
        return accessor(*this, args...);
    }

    template<typename... Ts>
    auto get_host_access(Ts... args) {
        return host_accessor(*this, args...);
    }

    template<typename Destination = std::nullptr_t>
    void set_final_data(Destination final_data = nullptr);

    void set_write_back(bool flag = true);

    bool is_sub_buffer() const;

    template<typename ReinterpretT, int ReinterpretDim>
    buffer<ReinterpretT, ReinterpretDim,
        typename std::allocator_traits<AllocatorT>::template rebind_alloc<ReinterpretT>>
    reinterpret(range<ReinterpretDim> reinterpret_range) const;

    template<typename ReinterpretT, int ReinterpretDim = Dimensions>
        requires(ReinterpretDim == 1 || (ReinterpretDim == Dimensions && sizeof(ReinterpretT) == sizeof(T)))
    buffer<ReinterpretT, ReinterpretDim,
        typename std::allocator_traits<AllocatorT>::template rebind_alloc<ReinterpretT>>
    reinterpret() const;

  private:
    template<typename>
    friend class detail::weak_ref;

    template<typename U, int D, typename A>
    friend U *simsycl::detail::get_buffer_data(sycl::buffer<U, D, A> &buf);

    using reference_type::state;

    buffer(std::shared_ptr<state_type> &&state) : reference_type(std::move(state)) {}
};

// Deduction guides
template<class InputIterator, class AllocatorT>
buffer(InputIterator, InputIterator, AllocatorT, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1, AllocatorT>;

template<class InputIterator>
buffer(InputIterator, InputIterator, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;

template<class T, int Dimensions, class AllocatorT>
buffer(const T *, const range<Dimensions> &, AllocatorT, const property_list & = {})
    -> buffer<T, Dimensions, AllocatorT>;

template<class T, int Dimensions>
buffer(const T *, const range<Dimensions> &, const property_list & = {}) -> buffer<T, Dimensions>;

template<typename Container, typename AllocatorT>
    requires(detail::is_container_v<Container, typename Container::value_type>)
buffer(Container &, AllocatorT, const property_list & = {}) -> buffer<typename Container::value_type, 1, AllocatorT>;

template<typename Container>
    requires(detail::is_container_v<Container, typename Container::value_type>)
buffer(Container &, const property_list & = {}) -> buffer<typename Container::value_type, 1>;

} // namespace simsycl::sycl

template<typename T, int Dimensions, typename AllocatorT>
class std::hash<simsycl::sycl::buffer<T, Dimensions, AllocatorT>>
    : std::hash<simsycl::detail::reference_type<simsycl::sycl::buffer<T, Dimensions, AllocatorT>,
          simsycl::detail::buffer_state<T, Dimensions, AllocatorT>>> {};

namespace simsycl::detail {

template<typename T, int Dimensions, typename AllocatorT>
T *get_buffer_data(sycl::buffer<T, Dimensions, AllocatorT> &buf) {
    return buf.state().buffer;
}

} // namespace simsycl::detail
