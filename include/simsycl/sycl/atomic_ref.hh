#pragma once

#include "atomic_fence.hh"
#include "enums.hh"

#include "../detail/utils.hh"

#include <cstdlib>
#include <utility>


namespace simsycl::detail {

using sycl::memory_order;
using sycl::memory_scope;

// Exposition only
template<memory_order ReadModifyWriteOrder>
struct memory_order_traits;

template<>
struct memory_order_traits<sycl::memory_order::relaxed> {
    static constexpr memory_order read_order = memory_order::relaxed;
    static constexpr memory_order write_order = memory_order::relaxed;
};

template<>
struct memory_order_traits<sycl::memory_order::acq_rel> {
    static constexpr memory_order read_order = memory_order::acquire;
    static constexpr memory_order write_order = memory_order::release;
};

template<>
struct memory_order_traits<memory_order::seq_cst> {
    static constexpr memory_order read_order = memory_order::seq_cst;
    static constexpr memory_order write_order = memory_order::seq_cst;
};

template<typename T, memory_order DefaultOrder, memory_scope DefaultScope, sycl::access::address_space AddressSpace>
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

    void store(T operand, memory_order order = default_write_order, memory_scope scope = default_scope) noexcept {
        m_ref = operand;
        atomic_fence(order, scope);
    }

    T operator=(T desired) noexcept {
        store(desired);
        return desired;
    }

    T load(memory_order order = default_read_order, memory_scope scope = default_scope) const noexcept {
        atomic_fence(order, scope);
        return m_ref;
    }

    operator T() const noexcept { return load(); }

    T exchange(
        T operand, memory_order order = default_read_modify_write_order, memory_scope scope = default_scope) noexcept {
        using std::swap;
        swap(m_ref, operand);
        atomic_fence(order, scope);
        return operand;
    }

    bool compare_exchange_weak(T &expected, T desired, memory_order success, memory_order failure,
        memory_scope scope = default_scope) noexcept {
        return compare_exchange(expected, desired, success, failure, scope);
    }

    bool compare_exchange_weak(T &expected, T desired, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        return compare_exchange(expected, desired, order, order, scope);
    }

    bool compare_exchange_strong(T &expected, T desired, memory_order success, memory_order failure,
        memory_scope scope = default_scope) noexcept {
        return compare_exchange(expected, desired, success, failure, scope);
    }

    bool compare_exchange_strong(T &expected, T desired, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        return compare_exchange(expected, desired, order, order, scope);
    }

  protected:
    T &m_ref;

  private:
    bool compare_exchange(
        T &expected, T desired, memory_order success, memory_order failure, memory_scope scope) noexcept {
        // TODO randomly fail a weak compare_exchange
        if(m_ref == expected) {
            m_ref = desired;
            atomic_fence(success, scope);
            return true;
        } else {
            expected = m_ref;
            atomic_fence(failure, scope);
            return false;
        }
    }
};

// The spec requires that atomic_ref<T> is only valid for types T that are 4 or 8 bytes in size
// (8 is only valid for devices with aspect::atomic64, but we can't check that at compile time)
template<typename T>
concept ValidAtomicSize = (sizeof(int) == 4 || sizeof(int) == 8);

// The spec provides an explicit list of which types are valid for atomic operations
template<typename T>
concept ValidAtomicIntType
    = (std::same_as<T, int> || std::same_as<T, unsigned int> || std::same_as<T, long> || std::same_as<T, unsigned long>
          || std::same_as<T, long long> || std::same_as<T, unsigned long long>)
    && ValidAtomicSize<T>;

template<typename T>
concept ValidAtomicFloatType = (std::same_as<T, float> || std::same_as<T, double>) && ValidAtomicSize<T>;
// TODO: for double we would also need to check if
//           "the code containing this sycl::atomic_ref was submitted to a device that has sycl::aspect::atomic64"
// that's quite challenging to implement as-is, since we would need to do this at runtime when operations are performed

template<typename T>
concept ValidAtomicPointerType
    = std::is_pointer_v<T> && std::is_object_v<std::remove_pointer_t<T>> && ValidAtomicSize<T>;

template<typename T>
concept ValidAtomicType = ValidAtomicIntType<T> || ValidAtomicFloatType<T> || ValidAtomicPointerType<T>;

} // namespace simsycl::detail

namespace simsycl::sycl {

template<detail::ValidAtomicType T, memory_order DefaultOrder, memory_scope DefaultScope,
    access::address_space AddressSpace = access::address_space::generic_space>
class atomic_ref : public detail::atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace> {
  private:
    using base = detail::atomic_ref_base<T, DefaultOrder, DefaultScope, AddressSpace>;

  public:
    using base::base;
    using base::operator=;
};

// Partial specialization for integral types
template<detail::ValidAtomicIntType Integral, memory_order DefaultOrder, memory_scope DefaultScope,
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

    using base::base;
    using base::operator=;

    Integral fetch_add(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref += operand;
        atomic_fence(order, scope);
        return original;
    }

    Integral fetch_sub(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref -= operand;
        atomic_fence(order, scope);
        return original;
    }

    Integral fetch_and(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref &= operand;
        atomic_fence(order, scope);
        return original;
    }

    Integral fetch_or(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref |= operand;
        atomic_fence(order, scope);
        return original;
    }

    Integral fetch_xor(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref ^= operand;
        atomic_fence(order, scope);
        return original;
    }

    Integral fetch_min(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref = detail::min(m_ref, operand);
        atomic_fence(order, scope);
        return original;
    }

    Integral fetch_max(Integral operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref = detail::max(m_ref, operand);
        atomic_fence(order, scope);
        return original;
    }

    Integral operator++(int) noexcept { return fetch_add(1); }
    Integral operator--(int) noexcept { return fetch_sub(1); }
    Integral operator++() noexcept { return fetch_add(1) + 1; }
    Integral operator--() noexcept { return fetch_sub(1) - 1; }
    Integral operator+=(Integral operand) noexcept { return fetch_add(operand) + operand; }
    Integral operator-=(Integral operand) noexcept { return fetch_sub(operand) - operand; }
    Integral operator&=(Integral operand) noexcept { return fetch_and(operand) & operand; }
    Integral operator|=(Integral operand) noexcept { return fetch_or(operand) | operand; }
    Integral operator^=(Integral operand) noexcept { return fetch_xor(operand) ^ operand; }

  private:
    using base::m_ref;
};

// Partial specialization for floating-point types
template<detail::ValidAtomicFloatType Floating, memory_order DefaultOrder, memory_scope DefaultScope,
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

    using base::base;
    using base::operator=;

    Floating fetch_add(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref += operand;
        atomic_fence(order, scope);
        return original;
    }

    Floating fetch_sub(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref -= operand;
        atomic_fence(order, scope);
        return original;
    }

    Floating fetch_min(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref = detail::min(m_ref, operand);
        atomic_fence(order, scope);
        return original;
    }

    Floating fetch_max(Floating operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref = detail::max(m_ref, operand);
        atomic_fence(order, scope);
        return original;
    }

    Floating operator+=(Floating operand) noexcept { return fetch_add(operand) + operand; }
    Floating operator-=(Floating operand) noexcept { return fetch_sub(operand) - operand; }

  private:
    using base::m_ref;
};

// Partial specialization for pointers
template<typename T, memory_order DefaultOrder, memory_scope DefaultScope, access::address_space AddressSpace>
class atomic_ref<T *, DefaultOrder, DefaultScope, AddressSpace>
    : public detail::atomic_ref_base<T *, DefaultOrder, DefaultScope, AddressSpace> {
  private:
    using base = detail::atomic_ref_base<T *, DefaultOrder, DefaultScope, AddressSpace>;

  public:
    using typename base::value_type;
    using difference_type = std::ptrdiff_t;

    using base::default_read_modify_write_order;
    using base::default_scope;

    using base::base;
    using base::operator=;

    T *fetch_add(difference_type operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref += operand;
        atomic_fence(order, scope);
        return original;
    }

    T *fetch_sub(difference_type operand, memory_order order = default_read_modify_write_order,
        memory_scope scope = default_scope) noexcept {
        const auto original = m_ref;
        m_ref -= operand;
        atomic_fence(order, scope);
        return original;
    }

    T *operator++(int) noexcept { return fetch_add(1); }
    T *operator--(int) noexcept { return fetch_sub(1); }
    T *operator++() noexcept { return fetch_add(1) + 1; }
    T *operator--() noexcept { return fetch_sub(1) - 1; }
    T *operator+=(difference_type operand) noexcept { return fetch_add(operand) + operand; }
    T *operator-=(difference_type operand) noexcept { return fetch_sub(operand) - operand; }

  private:
    using base::m_ref;
};

} // namespace simsycl::sycl
