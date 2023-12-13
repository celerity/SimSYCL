#pragma once

#include "binary_ops.hh"
#include "forward.hh"
#include "handler.hh"
#include "property.hh"

#include "../detail/check.hh"
#include "../detail/subscript.hh"

#include <span>


namespace simsycl::sycl::property::reduction {

struct initialize_to_identity {};

}; // namespace simsycl::sycl::property::reduction

namespace simsycl::sycl {

template <>
struct is_property<property::reduction::initialize_to_identity> : std::true_type {};

} // namespace simsycl::sycl

namespace simsycl::detail {

template <typename T, typename BinaryOperation, int Dimensions>
class reducer {
  public:
    using value_type = T;
    using binary_operation = BinaryOperation;
    static constexpr int dimensions = Dimensions;

    explicit reducer(T *value, BinaryOperation combiner) : m_dim0(value, combiner) {}

    reducer(const reducer &) = delete;
    reducer(reducer &&) = delete;
    reducer &operator=(const reducer &) = delete;
    reducer &operator=(reducer &&) = delete;

    decltype(auto) operator[](size_t index) { return subscript<Dimensions>(*this, index); }

    T identity() const
        requires(sycl::has_known_identity_v<BinaryOperation, T>)
    {
        return sycl::known_identity_v<BinaryOperation, T>;
    }

  private:
    template <int D, typename U, int S>
    friend decltype(auto) subscript(U &, sycl::id<D>, size_t);

    reducer<T, BinaryOperation, 0> m_dim0;

    reducer<T, BinaryOperation, 0> &operator[](sycl::id<Dimensions> index) {
        SIMSYCL_CHECK(index == sycl::id<Dimensions>{});
        return m_dim0;
    }
};

template <typename T, typename BinaryOperation>
class reducer<T, BinaryOperation, 0> {
  public:
    using value_type = T;
    using binary_operation = BinaryOperation;
    static constexpr int dimensions = 0;

    explicit reducer(T *value, BinaryOperation combiner) : m_value(value), m_combiner(combiner) {}

    reducer(const reducer &) = delete;
    reducer(reducer &&) = delete;
    reducer &operator=(const reducer &) = delete;
    reducer &operator=(reducer &&) = delete;

    reducer &combine(const T &partial) {
        *m_value = m_combiner(*m_value, partial);
        return *this;
    }

    T identity() const
        requires(sycl::has_known_identity_v<BinaryOperation, T>)
    {
        return sycl::known_identity_v<BinaryOperation, T>;
    }

    friend reducer &operator+=(reducer &lhs, const T &rhs)
        requires(std::is_same_v<BinaryOperation, sycl::plus<>> || std::is_same_v<BinaryOperation, sycl::plus<T>>)
    {
        *lhs.m_value += rhs;
        return lhs;
    }

    friend reducer &operator*=(reducer &lhs, const T &rhs)
        requires(
            std::is_same_v<BinaryOperation, sycl::multiplies<>> || std::is_same_v<BinaryOperation, sycl::multiplies<T>>)
    {
        *lhs.m_value *= rhs;
        return lhs;
    }

    friend reducer &operator&=(reducer &lhs, const T &rhs)
        requires(std::is_same_v<BinaryOperation, sycl::bit_and<>> || std::is_same_v<BinaryOperation, sycl::bit_and<T>>)
    {
        *lhs.m_value &= rhs;
        return lhs;
    }

    friend reducer &operator|=(reducer &lhs, const T &rhs)
        requires(std::is_same_v<BinaryOperation, sycl::bit_or<>> || std::is_same_v<BinaryOperation, sycl::bit_or<T>>)
    {
        *lhs.m_value |= rhs;
        return lhs;
    }

    friend reducer &operator^=(reducer &lhs, const T &rhs)
        requires(std::is_same_v<BinaryOperation, sycl::bit_xor<>> || std::is_same_v<BinaryOperation, sycl::bit_xor<T>>)
    {
        *lhs.m_value ^= rhs;
        return lhs;
    }

    friend reducer &operator++(reducer &lhs)
        requires(std::is_same_v<BinaryOperation, sycl::plus<>> || std::is_same_v<BinaryOperation, sycl::plus<T>>)
    {
        ++*lhs.m_value;
        return lhs;
    }

  private:
    T *m_value;
    BinaryOperation m_combiner;
};

template <typename T, typename BinaryOperation>
void begin_reduction(T *value, BinaryOperation /* combiner */, const std::type_identity_t<T> *explicit_identity,
    const sycl::property_list &prop_list) {
    const property_interface props(
        prop_list, property_compatibility<sycl::property::reduction::initialize_to_identity>{});
    if(props.has_property<sycl::property::reduction::initialize_to_identity>()) {
        if(explicit_identity != nullptr) {
            *value = *explicit_identity;
        } else if constexpr(sycl::has_known_identity_v<BinaryOperation, T>) {
            *value = sycl::known_identity_v<BinaryOperation, T>;
        } else {
            SIMSYCL_CHECK(false && "No identity provided for reduction");
        }
    }
}

} // namespace simsycl::detail

namespace simsycl::sycl {

template <class T, std::size_t Extent = std::dynamic_extent>
using span = std::span<T, Extent>;

// TODO in the spec, this simply accepts `typename BufferT` - is this more restrictive?
template <typename T, int Dimensions, typename AllocatorT, typename BinaryOperation>
auto reduction(buffer<T, Dimensions, AllocatorT> &vars, handler &cgh, BinaryOperation combiner,
    const property_list &prop_list = {}) {
    (void)cgh;
    SIMSYCL_CHECK(vars.get_range().size() == 1);
    T *value = detail::get_buffer_data(vars);
    detail::begin_reduction(value, combiner, nullptr, prop_list);
    return detail::reducer<T, BinaryOperation, Dimensions>(value, combiner);
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, BinaryOperation combiner, const property_list &prop_list = {}) {
    detail::begin_reduction(var, combiner, nullptr, prop_list);
    return detail::reducer<T, BinaryOperation, 0>(var, combiner);
}

template <typename T, size_t Extent, typename BinaryOperation>
    requires(Extent != std::dynamic_extent)
auto reduction(span<T, Extent> vars, BinaryOperation combiner, const property_list &prop_list = {});

// TODO in the spec, this simply accepts `typename BufferT` - is this more restrictive?
template <typename T, int Dimensions, typename AllocatorT, typename BinaryOperation>
auto reduction(buffer<T, Dimensions, AllocatorT> &vars, handler &cgh, const T &identity, BinaryOperation combiner,
    const property_list &prop_list = {}) {
    (void)cgh;
    SIMSYCL_CHECK(vars.get_range().size() == 1);
    T *value = detail::get_buffer_data(vars);
    detail::begin_reduction(value, combiner, &identity, prop_list);
    return detail::reducer<T, BinaryOperation, Dimensions>(value, combiner);
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, const T &identity, BinaryOperation combiner, const property_list &prop_list = {}) {
    detail::begin_reduction(var, combiner, identity, prop_list);
    return detail::reducer<T, BinaryOperation, 0>(var, combiner);
}

template <typename T, size_t Extent, typename BinaryOperation>
    requires(Extent != std::dynamic_extent)
auto reduction(span<T, Extent> vars, const T &identity, BinaryOperation combiner, const property_list &prop_list = {});

} // namespace simsycl::sycl
