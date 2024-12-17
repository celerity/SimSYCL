#pragma once

#include "context.hh"
#include "forward.hh"
#include "property.hh"

#include "../detail/allocation.hh"
#include "../detail/lock.hh"
#include "../detail/reference_type.hh"

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

    context get_context() const { return m_context; }

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

template<int Dimensions>
struct accessed_range {
    sycl::id<Dimensions> offset;
    sycl::range<Dimensions> range;
    sycl::access_mode mode;

    accessed_range(sycl::id<Dimensions> offset, sycl::range<Dimensions> range, sycl::access_mode mode)
        : offset(offset), range(range), mode(mode) {}

    friend bool operator==(const accessed_range &lhs, const accessed_range &rhs) = default;

    bool conflicts_with(const accessed_range &other) const {
        if(mode == sycl::access_mode::read && other.mode == sycl::access_mode::read) return false;

        for(int i = 0; i < Dimensions; ++i) {
            if(offset[i] < other.offset[i]) {
                if(offset[i] + range[i] <= other.offset[i]) { return false; }
            } else {
                if(other.offset[i] + other.range[i] <= offset[i]) { return false; }
            }
        }
        return true;
    }
};

// Base class for buffer_state necessary to keep a reference in accessor instances which do not know AllocatorT
template<int Dimensions>
struct buffer_access_validator {
    std::vector<accessed_range<Dimensions>> live_host_accesses;

    buffer_access_validator() = default;
    buffer_access_validator(const buffer_access_validator &) = delete;
    buffer_access_validator(buffer_access_validator &&) = delete;
    buffer_access_validator &operator=(const buffer_access_validator &) = delete;
    buffer_access_validator &operator=(buffer_access_validator &&) = delete;

    void begin_host_access(const detail::accessed_range<Dimensions> &range) { live_host_accesses.push_back(range); }

    void end_host_access(const detail::accessed_range<Dimensions> &range) {
        auto &live = live_host_accesses;
        live.erase(std::remove(live.begin(), live.end(), range), live.end());
    }

    void check_access_from_command_group(const detail::accessed_range<Dimensions> &range) const {
        for(const auto &live_range : live_host_accesses) {
            SIMSYCL_CHECK(!live_range.conflicts_with(range)
                && "Command group accessor overlaps with a live host accessor for the same buffer range, this is not "
                   "supported by SimSYCL unless both are read-only accesses");
        }
    }
};

template<typename T, int Dimensions, typename AllocatorT>
struct buffer_state {
    using write_back_fn = std::function<void(const T *, size_t)>;

    sycl::range<Dimensions> range;
    AllocatorT allocator;
    T *buffer;
    write_back_fn write_back;
    bool write_back_enabled;
    std::shared_ptr<const void> shared_host_ptr; // keep the std::shared_ptr host pointer alive
    mutable shared_value<buffer_access_validator<Dimensions>> validator;

    buffer_state(sycl::range<Dimensions> range, const AllocatorT &allocator = {}, const T *init_from = nullptr,
        write_back_fn write_back = {}, const std::shared_ptr<const void> &shared_host_ptr = nullptr)
        : range(range), allocator(allocator), buffer(this->allocator.allocate(range.size())), write_back(write_back),
          write_back_enabled(static_cast<bool>(write_back)), shared_host_ptr(shared_host_ptr) {
        if(init_from) {
            memcpy(buffer, init_from, range.size() * sizeof(T));
        } else {
            memset(buffer, static_cast<int>(detail::uninitialized_memory_pattern), range.size() * sizeof(T));
        }
    }

    template<typename InputIterator>
    buffer_state(InputIterator first, InputIterator last, const AllocatorT &allocator)
        : range(static_cast<size_t>(std::distance(first, last))), allocator(allocator),
          buffer(this->allocator.allocate(range.size())), write_back_enabled(false) {
        std::copy(first, last, buffer);
    }

    buffer_state(const buffer_state &) = delete;
    buffer_state(buffer_state &&) = delete;
    buffer_state &operator=(const buffer_state &) = delete;
    buffer_state &operator=(buffer_state &&) = delete;

    ~buffer_state() {
        if(write_back_enabled) {
            system_lock lock; // writeback must not overlap with command groups in other threads
            write_back(buffer, range.size());
        }
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
    using state_type = detail::buffer_state<std::remove_const_t<T>, Dimensions, AllocatorT>;
    using reference_type = detail::reference_type<buffer<T, Dimensions, AllocatorT>, state_type>;
    using write_back_fn = typename state_type::write_back_fn;
    using property_compatibility = detail::property_compatibility_with<buffer<T, Dimensions, AllocatorT>,
        property::buffer::use_host_ptr, property::buffer::use_mutex, property::buffer::context_bound>;

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
        requires(!std::is_const_v<T>)
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    buffer(
        T *host_data, const range<Dimensions> &buffer_range, AllocatorT allocator, const property_list &prop_list = {})
        requires(!std::is_const_v<T>)
        : property_interface(prop_list, property_compatibility()),
          reference_type(std::in_place, buffer_range, allocator, host_data, write_back_to(host_data)) {}

    buffer(const T *host_data, const range<Dimensions> &buffer_range, const property_list &prop_list = {})
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    buffer(const T *host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          reference_type(std::in_place, buffer_range, allocator, host_data) {}

    template<simsycl::detail::Container<T> Container>
        requires(Dimensions == 1)
    buffer(Container &container, AllocatorT allocator, const property_list &prop_list = {})
        : buffer(std::data(container), range(std::size(container)), allocator, prop_list) {}

    template<simsycl::detail::Container<T> Container>
        requires(Dimensions == 1)
    buffer(Container &container, const property_list &prop_list = {}) : buffer(container, AllocatorT(), prop_list) {}

    buffer(const std::shared_ptr<T> &host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {})
        requires(!std::is_const_v<T>)
        : property_interface(prop_list, property_compatibility()),
          reference_type(std::in_place, buffer_range, allocator, host_data.get(),
              write_back_to_if_non_const(host_data.get()), host_data) {}

    buffer(
        const std::shared_ptr<T> &host_data, const range<Dimensions> &buffer_range, const property_list &prop_list = {})
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    buffer(const std::shared_ptr<T[]> &host_data, const range<Dimensions> &buffer_range, AllocatorT allocator,
        const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          reference_type(std::in_place, buffer_range, allocator, host_data.get(),
              write_back_to_if_non_const(host_data.get()), host_data) {}

    buffer(const std::shared_ptr<T[]> &host_data, const range<Dimensions> &buffer_range,
        const property_list &prop_list = {})
        : buffer(host_data, buffer_range, AllocatorT(), prop_list) {}

    template<class InputIterator>
        requires(Dimensions == 1)
    buffer(InputIterator first, InputIterator last, AllocatorT allocator, const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          reference_type(std::in_place, first, last, allocator) {}

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
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL accessor<T, Dimensions, Mode, target::host_buffer> get_access() {
        return accessor<T, Dimensions, Mode, target::host_buffer>(*this);
    }

    template<access_mode Mode>
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL accessor<T, Dimensions, Mode, target::host_buffer> get_access(
        range<Dimensions> access_range, id<Dimensions> access_offset = {}) {
        return accessor<T, Dimensions, Mode, target::host_buffer>(*this, access_range, access_offset);
    }

    SIMSYCL_STOP_IGNORING_DEPRECATIONS

    template<typename... Ts>
    auto get_access(Ts &&...args) {
        return accessor(*this, std::forward<Ts>(args)...);
    }

    template<typename... Ts>
    auto get_host_access(Ts &&...args) {
        return host_accessor(*this, std::forward<Ts>(args)...);
    }

    template<typename Destination = std::nullptr_t>
    void set_final_data(Destination final_data = nullptr) {
        if constexpr(std::is_same_v<Destination, std::nullptr_t>) {
            state().write_back = {};
            state().write_back_enabled = false;
        } else {
            state().write_back = write_back_to(final_data);
            state().write_back_enabled = true;
        }
    }

    void set_write_back(bool flag = true) { state().write_back_enabled = state().write_back && flag; }

    bool is_sub_buffer() const {
        // sub-buffers are unimplemented
        return false;
    }

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

    template<typename U, int D, typename A>
    friend detail::buffer_access_validator<D> &detail::get_buffer_access_validator(
        const sycl::buffer<U, D, A> &buf, detail::system_lock &lock);

    using reference_type::state;

    static write_back_fn write_back_to(T out) {
        return [out](const T *buffer, size_t size) { return memcpy(out, buffer, size * sizeof(T)); };
    }

    static write_back_fn write_back_to(std::weak_ptr<T> out) {
        return [out](const T *buffer, size_t size) {
            if(const auto live = out.lock()) { return memcpy(live.get(), buffer, size * sizeof(T)); }
        };
    }

    template<typename OutputIterator>
    static write_back_fn write_back_to(OutputIterator out) {
        return [out](const T *buffer, size_t size) { return std::copy_n(buffer, size, out); };
    }

    static write_back_fn write_back_to_if_non_const(T *out) {
        if constexpr(!std::is_const_v<T>) {
            return write_back_to(out);
        } else {
            return {};
        }
    }

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
struct std::hash<simsycl::sycl::buffer<T, Dimensions, AllocatorT>>
    : std::hash<simsycl::detail::reference_type<simsycl::sycl::buffer<T, Dimensions, AllocatorT>,
          simsycl::detail::buffer_state<std::remove_const_t<T>, Dimensions, AllocatorT>>> {};

namespace simsycl::detail {

template<typename T, int Dimensions, typename AllocatorT>
T *get_buffer_data(sycl::buffer<T, Dimensions, AllocatorT> &buf) {
    return buf.state().buffer;
}

template<typename T, int Dimensions, typename AllocatorT>
buffer_access_validator<Dimensions> &get_buffer_access_validator(
    const sycl::buffer<T, Dimensions, AllocatorT> &buf, system_lock &lock) {
    return buf.state().validator.with(lock);
}

} // namespace simsycl::detail
