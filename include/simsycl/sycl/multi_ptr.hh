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

template<typename ElementType, access::address_space Space, access::decorated DecorateAddress>
class multi_ptr {
  public:
    static constexpr bool is_decorated = DecorateAddress == access::decorated::yes;
    static constexpr access::address_space address_space = Space;

    using value_type = ElementType;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference<value_type>;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;

    // Legacy has a different interface.
    static_assert(DecorateAddress != access::decorated::legacy);

    // Constructors
    multi_ptr();
    multi_ptr(const multi_ptr &);
    multi_ptr(multi_ptr &&);
    explicit multi_ptr(typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer);
    multi_ptr(std::nullptr_t);

    // Available only when:
    //   (Space == access::address_space::global_space ||
    //    Space == access::address_space::generic_space)
    template<int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
    multi_ptr(accessor<value_type, Dimensions, Mode, target::device, IsPlaceholder>);

    // Available only when:
    //   (Space == access::address_space::local_space ||
    //    Space == access::address_space::generic_space)
    template<int Dimensions>
    multi_ptr(local_accessor<ElementType, Dimensions>);

    // Deprecated
    // Available only when:
    //   (Space == access::address_space::local_space ||
    //    Space == access::address_space::generic_space)
    template<int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
    multi_ptr(accessor<value_type, Dimensions, Mode, target::local, IsPlaceholder>);

    // Assignment and access operators
    multi_ptr &operator=(const multi_ptr &);
    multi_ptr &operator=(multi_ptr &&);
    multi_ptr &operator=(std::nullptr_t);

    // Available only when:
    //   (Space == access::address_space::generic_space &&
    //    AS != access::address_space::constant_space)
    template<access::address_space AS, access::decorated IsDecorated>
    multi_ptr &operator=(const multi_ptr<value_type, AS, IsDecorated> &);

    // Available only when:
    //   (Space == access::address_space::generic_space &&
    //    AS != access::address_space::constant_space)
    template<access::address_space AS, access::decorated IsDecorated>
    multi_ptr &operator=(multi_ptr<value_type, AS, IsDecorated> &&);

    reference operator[](std::ptrdiff_t) const;

    reference operator*() const;
    pointer operator->() const;

    pointer get() const;
    pointer get_raw() const;
    pointer get_decorated() const;

    // Conversion to the underlying pointer type
    // Deprecated, get() should be used instead.
    operator pointer() const;

    // Cast to private_ptr
    // Available only when: (Space == access::address_space::generic_space)
    explicit operator multi_ptr<value_type, access::address_space::private_space, DecorateAddress>();

    // Cast to private_ptr
    // Available only when: (Space == access::address_space::generic_space)
    explicit operator multi_ptr<const value_type, access::address_space::private_space, DecorateAddress>() const;

    // Cast to global_ptr
    // Available only when: (Space == access::address_space::generic_space)
    explicit operator multi_ptr<value_type, access::address_space::global_space, DecorateAddress>();

    // Cast to global_ptr
    // Available only when: (Space == access::address_space::generic_space)
    explicit operator multi_ptr<const value_type, access::address_space::global_space, DecorateAddress>() const;

    // Cast to local_ptr
    // Available only when: (Space == access::address_space::generic_space)
    explicit operator multi_ptr<value_type, access::address_space::local_space, DecorateAddress>();

    // Cast to local_ptr
    // Available only when: (Space == access::address_space::generic_space)
    explicit operator multi_ptr<const value_type, access::address_space::local_space, DecorateAddress>() const;

    // Implicit conversion to a multi_ptr<void>.
    // Available only when: (!std::is_const_v<value_type>)
    template<access::decorated IsDecorated>
    operator multi_ptr<void, Space, IsDecorated>() const;

    // Implicit conversion to a multi_ptr<const void>.
    // Available only when: (std::is_const_v<value_type>)
    template<access::decorated IsDecorated>
    operator multi_ptr<const void, Space, IsDecorated>() const;

    // Implicit conversion to multi_ptr<const value_type, Space>.
    template<access::decorated IsDecorated>
    operator multi_ptr<const value_type, Space, IsDecorated>() const;

    // Implicit conversion to the non-decorated version of multi_ptr.
    // Available only when: (is_decorated == true)
    operator multi_ptr<value_type, Space, access::decorated::no>() const;

    // Implicit conversion to the decorated version of multi_ptr.
    // Available only when: (is_decorated == false)
    operator multi_ptr<value_type, Space, access::decorated::yes>() const;

    void prefetch(size_t num_elements) const;

    // Arithmetic operators
    friend multi_ptr &operator++(multi_ptr &mp) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(mp); }
    friend multi_ptr operator++(multi_ptr &mp, int) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(mp); }
    friend multi_ptr &operator--(multi_ptr &mp) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(mp); }
    friend multi_ptr operator--(multi_ptr &mp, int) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(mp); }
    friend multi_ptr &operator+=(multi_ptr &lhs, difference_type r) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, r); }
    friend multi_ptr &operator-=(multi_ptr &lhs, difference_type r) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, r); }
    friend multi_ptr operator+(const multi_ptr &lhs, difference_type r) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, r); }
    friend multi_ptr operator-(const multi_ptr &lhs, difference_type r) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, r); }
    friend reference operator*(const multi_ptr &lhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }

    friend bool operator==(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }
    friend bool operator!=(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }
    friend bool operator<(const multi_ptr &lhs, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs); }
    friend bool operator>(const multi_ptr &lhs, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs); }
    friend bool operator<=(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }
    friend bool operator>=(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }

    friend bool operator==(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator<(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator>(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
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
    multi_ptr();
    multi_ptr(const multi_ptr &);
    multi_ptr(multi_ptr &&);
    explicit multi_ptr(typename multi_ptr<VoidType, Space, access::decorated::yes>::pointer);
    multi_ptr(std::nullptr_t);

    // Available only when:
    //   (Space == access::address_space::global_space)
    template<typename ElementType, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
    multi_ptr(accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder>);

    // Available only when:
    //   (Space == access::address_space::local_space)
    template<typename ElementType, int Dimensions>
    multi_ptr(local_accessor<ElementType, Dimensions>);

    // Deprecated
    // Available only when:
    //   (Space == access::address_space::local_space)
    template<typename ElementType, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
    multi_ptr(accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder>);

    // Assignment operators
    multi_ptr &operator=(const multi_ptr &);
    multi_ptr &operator=(multi_ptr &&);
    multi_ptr &operator=(std::nullptr_t);

    pointer get() const;

    // Conversion to the underlying pointer type
    explicit operator pointer() const;

    // Explicit conversion to a multi_ptr<ElementType>
    // Available only when: (std::is_const_v<ElementType> || !std::is_const_v<VoidType>)
    template<typename ElementType>
    explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const;

    // Implicit conversion to the non-decorated version of multi_ptr.
    // Available only when: (is_decorated == true)
    operator multi_ptr<value_type, Space, access::decorated::no>() const;

    // Implicit conversion to the decorated version of multi_ptr.
    // Available only when: (is_decorated == false)
    operator multi_ptr<value_type, Space, access::decorated::yes>() const;

    // Implicit conversion to multi_ptr<const void, Space>
    operator multi_ptr<const void, Space, DecorateAddress>() const;

    friend bool operator==(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }
    friend bool operator!=(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }
    friend bool operator<(const multi_ptr &lhs, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs); }
    friend bool operator>(const multi_ptr &lhs, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs); }
    friend bool operator<=(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }
    friend bool operator>=(const multi_ptr &lhs, const multi_ptr &rhs) {
        SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs, rhs);
    }

    friend bool operator==(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator<(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator>(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }
    friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(lhs); }

    friend bool operator==(std::nullptr_t, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(rhs); }
    friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(rhs); }
    friend bool operator<(std::nullptr_t, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(rhs); }
    friend bool operator>(std::nullptr_t, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(rhs); }
    friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(rhs); }
    friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) { SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(rhs); }
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

// Deprecated, address_space_cast should be used instead.
template<typename ElementType, access::address_space Space, access::decorated DecorateAddress>
[[deprecated("use address_space_cast instead")]] multi_ptr<ElementType, Space, DecorateAddress> make_ptr(ElementType *);

template<access::address_space Space, access::decorated DecorateAddress, typename ElementType>
[[deprecated("use address_space_cast instead")]] multi_ptr<ElementType, Space, DecorateAddress> address_space_cast(
    ElementType *);

// Deduction guides
template<typename T, int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, Mode, target::device, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template<typename T, int Dimensions>
multi_ptr(local_accessor<T, Dimensions>) -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

template<typename ElementType, access::decorated IsDecorated = access::decorated::legacy>
using global_ptr = multi_ptr<ElementType, access::address_space::global_space, IsDecorated>;

template<typename ElementType, access::decorated IsDecorated = access::decorated::legacy>
using local_ptr = multi_ptr<ElementType, access::address_space::local_space, IsDecorated>;

// Deprecated in SYCL 2020
template<typename ElementType>

using constant_ptr [[deprecated]]
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

} // namespace simsycl::sycl
