#pragma once

#include "enums.hh"

#include <algorithm>
#include <cstdlib>
#include <utility>


namespace simsycl::detail {

using sycl::memory_order;
using sycl::memory_scope;

// Exposition only
template <memory_order ReadModifyWriteOrder>
struct memory_order_traits;

template <>
struct memory_order_traits<sycl::memory_order::relaxed> {
    static constexpr memory_order read_order = memory_order::relaxed;
    static constexpr memory_order write_order = memory_order::relaxed;
};

template <>
struct memory_order_traits<sycl::memory_order::acq_rel> {
    static constexpr memory_order read_order = memory_order::acquire;
    static constexpr memory_order write_order = memory_order::release;
};

template <>
struct memory_order_traits<memory_order::seq_cst> {
    static constexpr memory_order read_order = memory_order::seq_cst;
    static constexpr memory_order write_order = memory_order::seq_cst;
};

template <typename T, memory_order DefaultOrder, memory_scope DefaultScope, sycl::access::address_space AddressSpace>
class atomic_ref_base {
  public:
    using value_type = T;

    static constexpr bool is_always_lock_free = true;
    static constexpr size_t required_alignment = alignof(T);

    static constexpr memory_order default_read_order = memory_order_traits<DefaultOrder>::read_order;
    static constexpr memory_order default_write_order = memory_order_traits<DefaultOrder>::write_order;
    static constexpr memory_order default_read_modify_write_order = DefaultOrder;
    static constexpr memory_scope default_scope = DefaultScope;

    bool is_lock_free() const noexcept { return true; }

    explicit atomic_ref_base(T &ref) : m_ref(ref) {}
    atomic_ref_base(const atomic_ref_base &) noexcept = default;
    atomic_ref_base &operator=(const atomic_ref_base &) = delete;

    void store(T operand, memory_order order = default_write_order, memory_scope scope = default_scope) const noexcept {
        m_ref = operand;
        (void)order;
        (void)scope;
    }

    T operator=(T desired) const noexcept {
        m_ref = desired;
        return desired;
    }

    T load(memory_order order = default_read_order, memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        return m_ref;
    }

    operator T() const noexcept { return m_ref; }

    T exchange(T operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        using std::swap;
        (void)order;
        (void)scope;
        swap(m_ref, operand);
        return operand;
    }

    bool compare_exchange_weak(T &expected, T desired, memory_order success, memory_order failure,
        memory_scope scope = default_scope) const noexcept {
        (void)success;
        (void)failure;
        (void)scope;
        return compare_exchange(expected, desired);
    }

    bool compare_exchange_weak(T &expected, T desired, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        return compare_exchange(expected, desired);
    }

    bool compare_exchange_strong(T &expected, T desired, memory_order success, memory_order failure,
        memory_scope scope = default_scope) const noexcept {
        (void)success;
        (void)failure;
        (void)scope;
        return compare_exchange(expected, desired);
    }

    bool compare_exchange_strong(T &expected, T desired, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)scope;
        return compare_exchange(expected, desired);
    }

  protected:
    T &m_ref;

  private:
    bool compare_exchange(T &expected, T desired) {
        if(m_ref == expected) {
            m_ref = desired;
            return true;
        } else {
            expected = m_ref;
            return false;
        }
    }
};

} // namespace simsycl::detail

namespace simsycl::sycl {

template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
    access::address_space AddressSpace = access::address_space::generic_space>
class atomic_ref : public detail::atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace> {
  private:
    using base = detail::atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>;

  public:
    using base::base;
    using base::operator=;
};

// Partial specialization for integral types
template <std::integral Integral, memory_order DefaultOrder, memory_scope DefaultScope,
    access::address_space AddressSpace>
class atomic_ref<Integral, DefaultOrder, DefaultScope, AddressSpace>
    : public detail::atomic_ref_base<Integral, DefaultOrder, DefaultScope, AddressSpace> {
  private:
    using base = detail::atomic_ref_base<Integral, DefaultOrder, DefaultScope, AddressSpace>;

  public:
    using typename base::value_type;
    using difference_type = value_type;

    using base::default_read_modify_write_order;
    using base::default_scope;

    Integral fetch_add(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref += operand;
        return original;
    }

    Integral fetch_sub(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref -= operand;
        return original;
    }

    Integral fetch_and(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref &= operand;
        return original;
    }

    Integral fetch_or(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref |= operand;
        return original;
    }

    Integral fetch_xor(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref ^= operand;
        return original;
    }

    Integral fetch_min(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref = std::min(m_ref, operand);
        return original;
    }

    Integral fetch_max(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref = std::max(m_ref, operand);
        return original;
    }

    Integral operator++(int) const noexcept { return fetch_add(1); }
    Integral operator--(int) const noexcept { return fetch_sub(1); }
    Integral operator++() const noexcept { return fetch_add(1) + 1; }
    Integral operator--() const noexcept { return fetch_sub(1) - 1; }
    Integral operator+=(Integral operand) const noexcept { return fetch_add(operand) + operand; }
    Integral operator-=(Integral operand) const noexcept { return fetch_sub(operand) - operand; }
    Integral operator&=(Integral operand) const noexcept { return fetch_and(operand) & operand; }
    Integral operator|=(Integral operand) const noexcept { return fetch_and(operand) | operand; }
    Integral operator^=(Integral operand) const noexcept { return fetch_and(operand) ^ operand; }

  private:
    using base::m_ref;
};

// Partial specialization for floating-point types
template <std::floating_point Floating, memory_order DefaultOrder, memory_scope DefaultScope,
    access::address_space AddressSpace>
class atomic_ref<Floating, DefaultOrder, DefaultScope, AddressSpace>
    : public detail::atomic_ref_base<Floating, DefaultOrder, DefaultScope, AddressSpace> {
  private:
    using base = detail::atomic_ref_base<Floating, DefaultOrder, DefaultScope, AddressSpace>;

  public:
    using typename base::value_type;
    using difference_type = value_type;

    using base::default_read_modify_write_order;
    using base::default_scope;

    Floating fetch_add(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref += operand;
        return original;
    }

    Floating fetch_sub(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref -= operand;
        return original;
    }

    Floating fetch_min(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref = std::min(m_ref, operand);
        return original;
    }

    Floating fetch_max(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref = std::max(m_ref, operand);
        return original;
    }

    Floating operator+=(Floating operand) const noexcept { return fetch_add(operand) + operand; }
    Floating operator-=(Floating operand) const noexcept { return fetch_sub(operand) - operand; }

  private:
    using base::m_ref;
};

// Partial specialization for pointers
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope, access::address_space AddressSpace>
class atomic_ref<T *, DefaultOrder, DefaultScope, AddressSpace>
    : public detail::atomic_ref_base<T *, DefaultOrder, DefaultScope, AddressSpace> {
  private:
    using base = detail::atomic_ref_base<T *, DefaultOrder, DefaultScope, AddressSpace>;

  public:
    using typename base::value_type;
    using difference_type = value_type;

    using base::default_read_modify_write_order;
    using base::default_scope;

    T *fetch_add(difference_type operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref += operand;
        return original;
    }

    T *fetch_sub(difference_type operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) const noexcept {
        (void)order;
        (void)scope;
        const auto original = m_ref;
        m_ref -= operand;
        return original;
    }

    T *operator++(int) const noexcept { return fetch_add(1); }
    T *operator--(int) const noexcept { return fetch_sub(1); }
    T *operator++() const noexcept { return fetch_add(1) + 1; }
    T *operator--() const noexcept { return fetch_sub(1) - 1; }
    T *operator+=(difference_type operand) const noexcept { return fetch_add(operand) + operand; }
    T *operator-=(difference_type operand) const noexcept { return fetch_sub(operand) - operand; }

  private:
    using base::m_ref;
};

} // namespace simsycl::sycl
