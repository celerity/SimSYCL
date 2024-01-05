#pragma once

#include "enums.hh"
#include "forward.hh"

#include <iterator>
#include <type_traits>

namespace simsycl::detail {

// helper to allow specializing for void* and const void* simultaneously
template<typename VoidType>
class void_type;

} // namespace simsycl::detail

namespace simsycl::sycl {

template<typename T>
struct remove_decoration {
    using type = T;
};

template<typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

// target::local, ...
SIMSYCL_START_IGNORING_DEPRECATIONS

template<typename ElementType, access::address_space Space, access::decorated DecorateAddress>
class multi_ptr {
  public:
    static constexpr bool is_decorated = DecorateAddress == access::decorated::yes;
    static constexpr access::address_space address_space = Space;

    using value_type = ElementType;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference_t<value_type>;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;

    // Legacy has a different interface.
    static_assert(DecorateAddress != access::decorated::legacy);

    // Constructors
    constexpr multi_ptr() : m_ptr(nullptr) {}
    multi_ptr(const multi_ptr &) = default;
    multi_ptr(multi_ptr &&) = default;
    constexpr explicit multi_ptr(pointer ptr) : m_ptr(ptr) {}
    constexpr multi_ptr(std::nullptr_t /* nullptr */) : m_ptr(nullptr) {}

    template<int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
        requires(Space == access::address_space::global_space || Space == access::address_space::generic_space)
    multi_ptr(accessor<value_type, Dimensions, Mode, target::device, IsPlaceholder> acc) : m_ptr(acc.get_pointer()) {}

    template<int Dimensions>
        requires(Space == access::address_space::local_space || Space == access::address_space::generic_space)
    multi_ptr(local_accessor<ElementType, Dimensions> acc) : m_ptr(acc.get_pointer()) {}

    template<int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
        requires(Space == access::address_space::local_space || Space == access::address_space::generic_space)
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL multi_ptr(
        accessor<value_type, Dimensions, Mode, target::local, IsPlaceholder> acc)
        : m_ptr(acc.get_pointer()) {}

    // Assignment and access operators
    multi_ptr &operator=(const multi_ptr &) = default;
    multi_ptr &operator=(multi_ptr &&) = default;

    multi_ptr &operator=(std::nullptr_t) {
        m_ptr = nullptr;
        return *this;
    }

    template<access::address_space AS, access::decorated IsDecorated>
        requires(Space == access::address_space::generic_space && AS != access::address_space::constant_space)
    multi_ptr &operator=(const multi_ptr<value_type, AS, IsDecorated> &other) {
        m_ptr = other.m_ptr;
    }

    template<access::address_space AS, access::decorated IsDecorated>
        requires(Space == access::address_space::generic_space && AS != access::address_space::constant_space)
    multi_ptr &operator=(multi_ptr<value_type, AS, IsDecorated> &&other) {
        m_ptr = other.m_ptr;
    }

    reference operator[](std::ptrdiff_t i) const { return m_ptr[i]; }

    reference operator*() const { return *m_ptr; }
    pointer operator->() const { return m_ptr; }

    pointer get() const { return m_ptr; }
    pointer get_raw() const { return m_ptr; }
    pointer get_decorated() const { return m_ptr; }

    // Conversion to the underlying pointer type
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL operator pointer() const { return m_ptr; }

    // Cast to private_ptr
    explicit operator multi_ptr<value_type, access::address_space::private_space, DecorateAddress>()
        requires(Space == access::address_space::generic_space)
    {
        return multi_ptr<value_type, access::address_space::private_space, DecorateAddress>{m_ptr};
    }

    // Cast to private_ptr
    explicit operator multi_ptr<const value_type, access::address_space::private_space, DecorateAddress>() const
        requires(Space == access::address_space::generic_space)
    {
        return multi_ptr<const value_type, access::address_space::private_space, DecorateAddress>{m_ptr};
    }

    // Cast to global_ptr
    explicit operator multi_ptr<value_type, access::address_space::global_space, DecorateAddress>()
        requires(Space == access::address_space::generic_space)
    {
        return multi_ptr<value_type, access::address_space::global_space, DecorateAddress>{m_ptr};
    }

    // Cast to global_ptr
    explicit operator multi_ptr<const value_type, access::address_space::global_space, DecorateAddress>() const
        requires(Space == access::address_space::generic_space)
    {
        return multi_ptr<const value_type, access::address_space::global_space, DecorateAddress>{m_ptr};
    }

    // Cast to local_ptr
    explicit operator multi_ptr<value_type, access::address_space::local_space, DecorateAddress>()
        requires(Space == access::address_space::generic_space)
    {
        return multi_ptr<value_type, access::address_space::local_space, DecorateAddress>{m_ptr};
    }

    // Cast to local_ptr
    explicit operator multi_ptr<const value_type, access::address_space::local_space, DecorateAddress>() const
        requires(Space == access::address_space::generic_space)
    {
        return multi_ptr<const value_type, access::address_space::local_space, DecorateAddress>{m_ptr};
    }

    // Implicit conversion to a multi_ptr<void>.
    template<access::decorated IsDecorated>
    operator multi_ptr<void, Space, IsDecorated>() const
        requires(!std::is_const_v<value_type>)
    {
        return multi_ptr<void, Space, IsDecorated>{m_ptr};
    }

    // Implicit conversion to a multi_ptr<const void>.
    template<access::decorated IsDecorated>
    operator multi_ptr<const void, Space, IsDecorated>() const
        requires(std::is_const_v<value_type>)
    {
        return multi_ptr<const void, Space, IsDecorated>{m_ptr};
    }

    // Implicit conversion to multi_ptr<const value_type, Space>.
    template<access::decorated IsDecorated>
    operator multi_ptr<const value_type, Space, IsDecorated>() const {
        return multi_ptr<const value_type, Space, IsDecorated>{m_ptr};
    }

    // Implicit conversion to the non-decorated version of multi_ptr.
    operator multi_ptr<value_type, Space, access::decorated::no>() const
        requires is_decorated
    {
        return multi_ptr<value_type, Space, access::decorated::no>{m_ptr};
    }

    // Implicit conversion to the decorated version of multi_ptr.
    operator multi_ptr<value_type, Space, access::decorated::yes>() const
        requires(!is_decorated)
    {
        return multi_ptr<value_type, Space, access::decorated::yes>{m_ptr};
    }

    void prefetch(size_t num_elements) const { (void)num_elements; }

    // Arithmetic operators

    friend multi_ptr &operator++(multi_ptr &mp) {
        ++mp.m_ptr;
        return mp;
    }

    friend multi_ptr operator++(multi_ptr &mp, int) { return multi_ptr{mp.m_ptr++}; }

    friend multi_ptr &operator--(multi_ptr &mp) {
        --mp.m_ptr;
        return mp;
    }

    friend multi_ptr operator--(multi_ptr &mp, int) { return multi_ptr{mp.m_ptr--}; }

    friend multi_ptr &operator+=(multi_ptr &lhs, difference_type r) {
        lhs.m_ptr += r;
        return lhs;
    }

    friend multi_ptr &operator-=(multi_ptr &lhs, difference_type r) {
        lhs.m_ptr -= r;
        return lhs;
    }

    friend multi_ptr operator+(const multi_ptr &lhs, difference_type r) { return multi_ptr{lhs.m_ptr + r}; }
    friend multi_ptr operator-(const multi_ptr &lhs, difference_type r) { return multi_ptr{lhs.m_ptr - r}; }

    // Spec error: conflicts with operator* above
    // friend reference operator*(const multi_ptr &lhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }

    friend bool operator==(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr == rhs.m_ptr; }
    friend bool operator!=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr != rhs.m_ptr; }
    friend bool operator<(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr < rhs.m_ptr; }
    friend bool operator>(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr > rhs.m_ptr; }
    friend bool operator<=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr <= rhs.m_ptr; }
    friend bool operator>=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr >= rhs.m_ptr; }

    friend bool operator==(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr == nullptr; }
    friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr != nullptr; }
    friend bool operator<(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr < nullptr; }
    friend bool operator>(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr > nullptr; }
    friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr <= nullptr; }
    friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr >= nullptr; }

  private:
    template<typename, access::address_space, access::decorated>
    friend class multi_ptr;

    pointer m_ptr = nullptr;
};

// specialization helper
template<typename VoidType, access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<simsycl::detail::void_type<VoidType>, Space, DecorateAddress> {
  public:
    static constexpr bool is_decorated = DecorateAddress == access::decorated::yes;
    static constexpr access::address_space address_space = Space;

    using value_type = VoidType;
    using pointer = std::add_pointer_t<value_type>;
    using difference_type = std::ptrdiff_t;

    static_assert(std::is_same_v<remove_decoration_t<pointer>, std::add_pointer_t<value_type>>);
    // Legacy has a different interface.
    static_assert(DecorateAddress != access::decorated::legacy);

    // Constructors
    constexpr multi_ptr() : m_ptr(nullptr) {}
    multi_ptr(const multi_ptr &) = default;
    multi_ptr(multi_ptr &&) = default;
    constexpr explicit multi_ptr(pointer ptr): m_ptr(ptr) {}
    constexpr multi_ptr(std::nullptr_t /* nullptr */) : m_ptr(nullptr) {}

    template<typename ElementType, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
        requires(Space == access::address_space::global_space)
    multi_ptr(accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder> acc) : m_ptr(acc.get_pointer()) {}

    template<typename ElementType, int Dimensions>
        requires(Space == access::address_space::local_space)
    multi_ptr(local_accessor<ElementType, Dimensions> acc) : m_ptr(acc.get_pointer()) {}

    template<typename ElementType, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
        requires(Space == access::address_space::local_space)
    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL multi_ptr(
        accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder> acc)
        : m_ptr(acc.get_pointer()) {}

    // Assignment operators
    multi_ptr &operator=(const multi_ptr &) = default;
    multi_ptr &operator=(multi_ptr &&) = default;

    multi_ptr &operator=(std::nullptr_t) {
        m_ptr = nullptr;
        return *this;
    }

    pointer get() const { return m_ptr; }

    // Conversion to the underlying pointer type
    explicit operator pointer() const { return m_ptr; }

    // Explicit conversion to a multi_ptr<ElementType>
    template<typename ElementType>
        requires(std::is_const_v<ElementType> || !std::is_const_v<VoidType>)
    explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const {
        return multi_ptr<ElementType, Space, DecorateAddress>{m_ptr};
    }

    // Implicit conversion to the non-decorated version of multi_ptr.
    operator multi_ptr<value_type, Space, access::decorated::no>() const
        requires is_decorated
    {
        return multi_ptr<value_type, Space, access::decorated::no>{m_ptr};
    }

    // Implicit conversion to the decorated version of multi_ptr.
    operator multi_ptr<value_type, Space, access::decorated::yes>() const
        requires(!is_decorated)
    {
        return multi_ptr<value_type, Space, access::decorated::yes>{m_ptr};
    }

    // Implicit conversion to multi_ptr<const void, Space>
    operator multi_ptr<const void, Space, DecorateAddress>() const {
        return multi_ptr<const void, Space, DecorateAddress>{m_ptr};
    }

    friend bool operator==(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr == rhs.m_ptr; }
    friend bool operator!=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr != rhs.m_ptr; }
    friend bool operator<(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr < rhs.m_ptr; }
    friend bool operator>(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr > rhs.m_ptr; }
    friend bool operator<=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr <= rhs.m_ptr; }
    friend bool operator>=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr >= rhs.m_ptr; }

    friend bool operator==(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr == nullptr; }
    friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr != nullptr; }
    friend bool operator<(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr < nullptr; }
    friend bool operator>(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr > nullptr; }
    friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr <= nullptr; }
    friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr >= nullptr; }

    friend bool operator==(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr == nullptr; }
    friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr != nullptr; }
    friend bool operator<(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr < nullptr; }
    friend bool operator>(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr > nullptr; }
    friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr <= nullptr; }
    friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr >= nullptr; }

  private:
    template<typename, access::address_space, access::decorated>
    friend class multi_ptr;

    pointer m_ptr = nullptr;
};

template<access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<void, Space, DecorateAddress>
    : public multi_ptr<simsycl::detail::void_type<void>, Space, DecorateAddress> {
    using multi_ptr<simsycl::detail::void_type<void>, Space, DecorateAddress>::multi_ptr;
};

template<access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<const void, Space, DecorateAddress>
    : public multi_ptr<simsycl::detail::void_type<const void>, Space, DecorateAddress> {
    using multi_ptr<simsycl::detail::void_type<const void>, Space, DecorateAddress>::multi_ptr;
};

// Legacy interface, inherited from 1.2.1.
template<typename ElementType, access::address_space Space>
class SIMSYCL_DETAIL_DEPRECATED_IN_SYCL multi_ptr<ElementType, Space, access::decorated::legacy> {
  public:
    using value_type = ElementType;
    using element_type = ElementType;
    using difference_type = std::ptrdiff_t;

    // Implementation defined pointer and reference types that correspond to
    // SYCL/OpenCL interoperability types for OpenCL C functions.
    using pointer_t = typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
    using const_pointer_t = typename multi_ptr<const ElementType, Space, access::decorated::yes>::pointer;
    using reference_t = typename multi_ptr<ElementType, Space, access::decorated::yes>::reference;
    using const_reference_t = typename multi_ptr<const ElementType, Space, access::decorated::yes>::reference;

    static constexpr access::address_space address_space = Space;

    // Constructors
    constexpr multi_ptr() : m_ptr(nullptr) {}
    multi_ptr(const multi_ptr &) = default;
    multi_ptr(multi_ptr &&) = default;
    constexpr multi_ptr(pointer_t ptr) : m_ptr(ptr) {}
    constexpr multi_ptr(std::nullptr_t /* nullptr */) : m_ptr(nullptr) {}
    ~multi_ptr() = default;

    // Assignment and access operators
    multi_ptr &operator=(const multi_ptr &) = default;
    multi_ptr &operator=(multi_ptr &&) = default;
    multi_ptr &operator=(pointer_t ptr) { m_ptr = ptr; }
    multi_ptr &operator=(std::nullptr_t /* nullptr */) { m_ptr = nullptr; }

    ElementType *operator->() const { return m_ptr; }

    template<typename AccDataT, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
        requires((Space == access::address_space::global_space || Space == access::address_space::generic_space)
            && (std::is_same_v<std::remove_const_t<ElementType>, std::remove_const_t<AccDataT>>)
            && (std::is_const_v<ElementType>
                || !std::is_const_v<
                    typename accessor<AccDataT, Dimensions, Mode, target::device, IsPlaceholder>::value_type>))
    multi_ptr(accessor<AccDataT, Dimensions, Mode, target::device, IsPlaceholder> acc) : m_ptr(acc.get_pointer()) {}

    template<typename AccDataT, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
        requires((Space == access::address_space::local_space || Space == access::address_space::generic_space)
            && (std::is_same_v<std::remove_const_t<ElementType>, std::remove_const_t<AccDataT>>)
            && (std::is_const_v<ElementType> || !std::is_const_v<AccDataT>))
    multi_ptr(accessor<AccDataT, Dimensions, Mode, target::local, IsPlaceholder> acc) : m_ptr(acc.get_pointer()) {}

    template<typename AccDataT, int Dimensions>
        requires(Space == access::address_space::local_space || Space == access::address_space::generic_space)
        && (std::is_same_v<std::remove_const_t<ElementType>, std::remove_const_t<AccDataT>>)
        && (std::is_const_v<ElementType> || !std::is_const_v<AccDataT>)
    multi_ptr(local_accessor<AccDataT, Dimensions> acc) : m_ptr(acc.get_pointer()) {}

    template<int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
    multi_ptr(accessor<ElementType, Dimensions, Mode, target::constant_buffer, IsPlaceholder> acc)
        requires(Space == access::address_space::constant_space)
        : m_ptr(acc.get_pointer()) {}

    // Returns the underlying OpenCL C pointer
    pointer_t get() const { return m_ptr; }

    std::add_pointer_t<value_type> get_raw() const { return m_ptr; }

    pointer_t get_decorated() const { return m_ptr; }

    // Implicit conversion to the underlying pointer type
    operator ElementType *() const { return m_ptr; }

    // Implicit conversion to a multi_ptr<void>
    operator multi_ptr<void, Space, access::decorated::legacy>() const
        requires(!std::is_const_v<ElementType>)
    {
        return multi_ptr<void, Space, access::decorated::legacy>{m_ptr};
    }

    // Implicit conversion to a multi_ptr<const void>
    operator multi_ptr<const void, Space, access::decorated::legacy>() const
        requires(std::is_const_v<ElementType>)
    {
        return multi_ptr<const void, Space, access::decorated::legacy>{m_ptr};
    }

    // Implicit conversion to multi_ptr<const ElementType, Space>
    operator multi_ptr<const ElementType, Space, access::decorated::legacy>() const {
        return multi_ptr<const ElementType, Space, access::decorated::legacy>{m_ptr};
    }

    // Arithmetic operators

    friend multi_ptr &operator++(multi_ptr &mp) {
        ++mp.m_ptr;
        return mp;
    }

    friend multi_ptr operator++(multi_ptr &mp, int) { return multi_ptr{mp.m_ptr++}; }

    friend multi_ptr &operator--(multi_ptr &mp) {
        --mp.m_ptr;
        return mp;
    }

    friend multi_ptr operator--(multi_ptr &mp, int) { return multi_ptr{mp.m_ptr--}; }

    friend multi_ptr &operator+=(multi_ptr &lhs, difference_type r) {
        lhs.m_ptr += r;
        return lhs;
    }
    friend multi_ptr &operator-=(multi_ptr &lhs, difference_type r) {
        lhs.m_ptr -= r;
        return lhs;
    }
    friend multi_ptr operator+(const multi_ptr &lhs, difference_type r) { return multi_ptr{lhs.m_ptr + r}; }
    friend multi_ptr operator-(const multi_ptr &lhs, difference_type r) { return multi_ptr{lhs.m_ptr - r}; }

    void prefetch(size_t num_elements) const { (void)num_elements; }

    friend bool operator==(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr == rhs.m_ptr; }
    friend bool operator!=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr != rhs.m_ptr; }
    friend bool operator<(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr < rhs.m_ptr; }
    friend bool operator>(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr > rhs.m_ptr; }
    friend bool operator<=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr <= rhs.m_ptr; }
    friend bool operator>=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr >= rhs.m_ptr; }

    friend bool operator==(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr == nullptr; }
    friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr != nullptr; }
    friend bool operator<(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr < nullptr; }
    friend bool operator>(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr > nullptr; }
    friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr <= nullptr; }
    friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr >= nullptr; }

    friend bool operator==(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr == nullptr; }
    friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr != nullptr; }
    friend bool operator<(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr < nullptr; }
    friend bool operator>(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr > nullptr; }
    friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr <= nullptr; }
    friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr >= nullptr; }

  private:
    template<typename, access::address_space, access::decorated>
    friend class multi_ptr;

    pointer_t m_ptr = nullptr;
};

// Legacy interface, inherited from 1.2.1.
template<typename VoidType, access::address_space Space>
class SIMSYCL_DETAIL_DEPRECATED_IN_SYCL
    multi_ptr<simsycl::detail::void_type<VoidType>, Space, access::decorated::legacy> {
  public:
    using value_type = VoidType;
    using element_type = VoidType;
    using difference_type = std::ptrdiff_t;

    // Implementation defined pointer types that correspond to
    // SYCL/OpenCL interoperability types for OpenCL C functions
    using pointer_t = typename multi_ptr<VoidType, Space, access::decorated::yes>::pointer;
    using const_pointer_t = typename multi_ptr<const VoidType, Space, access::decorated::yes>::pointer;

    static constexpr access::address_space address_space = Space;

    // Constructors
    constexpr multi_ptr() : m_ptr(nullptr) {}
    multi_ptr(const multi_ptr &) = default;
    multi_ptr(multi_ptr &&) = default;
    constexpr multi_ptr(pointer_t ptr) : m_ptr(ptr) {}
    constexpr multi_ptr(std::nullptr_t /* nullptr */) : m_ptr(nullptr) {}
    ~multi_ptr() = default;

    // Assignment operators
    multi_ptr &operator=(const multi_ptr &) = default;
    multi_ptr &operator=(multi_ptr &&) = default;
    multi_ptr &operator=(pointer_t ptr) { m_ptr = ptr; }
    multi_ptr &operator=(std::nullptr_t /* nullptr */) { m_ptr = nullptr; }

    template<typename ElementType, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
        requires(Space == access::address_space::global_space || Space == access::address_space::generic_space)
        && (std::is_const_v<VoidType>
            || !std::is_const_v<
                typename accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder>::value_type>)
    multi_ptr(accessor<ElementType, Dimensions, Mode, target::device> acc) : m_ptr(acc.get_pointer()) {}

    template<typename ElementType, int Dimensions, access_mode Mode>
        requires(Space == access::address_space::local_space || Space == access::address_space::generic_space)
        && (std::is_const_v<VoidType> || !std::is_const_v<ElementType>)
    multi_ptr(accessor<ElementType, Dimensions, Mode, target::local> acc) : m_ptr(acc.get_pointer()) {}

    template<typename AccDataT, int Dimensions>
        requires((Space == access::address_space::local_space || Space == access::address_space::generic_space)
            && (std::is_const_v<VoidType> || !std::is_const_v<element_type>))
    multi_ptr(local_accessor<AccDataT, Dimensions> acc) : m_ptr(acc.get_pointer()) {}

    template<typename ElementType, int Dimensions, access_mode Mode>
    multi_ptr(accessor<ElementType, Dimensions, Mode, target::constant_buffer> acc)
        requires(Space == access::address_space::constant_space)
        : m_ptr(acc.get_pointer()) {}

    // Returns the underlying OpenCL C pointer
    pointer_t get() const { return m_ptr; }

    std::add_pointer_t<value_type> get_raw() const { return m_ptr; }

    pointer_t get_decorated() const { return m_ptr; }

    // Implicit conversion to the underlying pointer type
    operator VoidType *() const { return m_ptr; }

    // Explicit conversion to a multi_ptr<ElementType>
    template<typename ElementType>
        requires(!std::is_const_v<VoidType> || std::is_const_v<ElementType>)
    explicit operator multi_ptr<ElementType, Space, access::decorated::legacy>() const {
        return multi_ptr<ElementType, Space, access::decorated::legacy>{m_ptr};
    }

    // Implicit conversion to multi_ptr<const void, Space>
    operator multi_ptr<const void, Space, access::decorated::legacy>() const {
        return multi_ptr<const void, Space, access::decorated::legacy>{m_ptr};
    }

    friend bool operator==(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr == rhs.m_ptr; }
    friend bool operator!=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr != rhs.m_ptr; }
    friend bool operator<(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr < rhs.m_ptr; }
    friend bool operator>(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr > rhs.m_ptr; }
    friend bool operator<=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr <= rhs.m_ptr; }
    friend bool operator>=(const multi_ptr &lhs, const multi_ptr &rhs) { return lhs.m_ptr >= rhs.m_ptr; }

    friend bool operator==(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr == nullptr; }
    friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr != nullptr; }
    friend bool operator<(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr < nullptr; }
    friend bool operator>(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr > nullptr; }
    friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr <= nullptr; }
    friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) { return lhs.m_ptr >= nullptr; }

    friend bool operator==(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr == nullptr; }
    friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr != nullptr; }
    friend bool operator<(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr < nullptr; }
    friend bool operator>(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr > nullptr; }
    friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr <= nullptr; }
    friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) { return rhs.m_ptr >= nullptr; }

  private:
    template<typename, access::address_space, access::decorated>
    friend class multi_ptr;

    pointer_t m_ptr = nullptr;
};

// need to specialize separately for void + decorated::legacy to avoid ambiguity

template<access::address_space Space>
class multi_ptr<void, Space, access::decorated::legacy>
    : public multi_ptr<simsycl::detail::void_type<void>, Space, access::decorated::legacy> {
    using multi_ptr<simsycl::detail::void_type<void>, Space, access::decorated::legacy>::multi_ptr;
};

template<access::address_space Space>
class multi_ptr<const void, Space, access::decorated::legacy>
    : public multi_ptr<simsycl::detail::void_type<const void>, Space, access::decorated::legacy> {
    using multi_ptr<simsycl::detail::void_type<const void>, Space, access::decorated::legacy>::multi_ptr;
};

// Deprecated, address_space_cast should be used instead.
template<typename ElementType, access::address_space Space, access::decorated DecorateAddress>
SIMSYCL_DETAIL_DEPRECATED_IN_SYCL_V("use address_space_cast instead")
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(ElementType *ptr) {
    return {ptr};
}

template<access::address_space Space, access::decorated DecorateAddress, typename ElementType>
SIMSYCL_DETAIL_DEPRECATED_IN_SYCL_V("use address_space_cast instead")
multi_ptr<ElementType, Space, DecorateAddress> address_space_cast(ElementType *ptr) {
    return {ptr};
}


// Deduction guides
template<typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read, target::device, IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::global_space, access::decorated::no>;

template<typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::write, target::device, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template<typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read_write, target::device, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template<typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read, target::constant_buffer, IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::constant_space, access::decorated::no>;

template<typename T, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, Mode, target::local, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

template<typename T, int Dimensions>
multi_ptr(local_accessor<T, Dimensions>) -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;


template<typename ElementType, access::decorated IsDecorated = access::decorated::legacy>
using global_ptr = multi_ptr<ElementType, access::address_space::global_space, IsDecorated>;

template<typename ElementType, access::decorated IsDecorated = access::decorated::legacy>
using local_ptr = multi_ptr<ElementType, access::address_space::local_space, IsDecorated>;

template<typename ElementType>
using constant_ptr SIMSYCL_DETAIL_DEPRECATED_IN_SYCL
    = multi_ptr<ElementType, access::address_space::constant_space, access::decorated::legacy>;

template<typename ElementType, access::decorated IsDecorated = access::decorated::legacy>
using private_ptr = multi_ptr<ElementType, access::address_space::private_space, IsDecorated>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes non-decorated pointer while keeping the address space information internally.

template<typename ElementType>
using raw_global_ptr = multi_ptr<ElementType, access::address_space::global_space, access::decorated::no>;

template<typename ElementType>
using raw_local_ptr = multi_ptr<ElementType, access::address_space::local_space, access::decorated::no>;

template<typename ElementType>
using raw_private_ptr = multi_ptr<ElementType, access::address_space::private_space, access::decorated::no>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes decorated pointer.
template<typename ElementType>
using decorated_global_ptr = multi_ptr<ElementType, access::address_space::global_space, access::decorated::yes>;

template<typename ElementType>
using decorated_local_ptr = multi_ptr<ElementType, access::address_space::local_space, access::decorated::yes>;

template<typename ElementType>
using decorated_private_ptr = multi_ptr<ElementType, access::address_space::private_space, access::decorated::yes>;

SIMSYCL_STOP_IGNORING_DEPRECATIONS

} // namespace simsycl::sycl
