#pragma once

#include "enums.hh"
#include "forward.hh"
#include "type_traits.hh"

#include "../detail/check.hh"
#include "../detail/utils.hh"

#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <type_traits>


namespace simsycl::sycl {

struct elem {
    static constexpr int x = 0;
    static constexpr int y = 1;
    static constexpr int z = 2;
    static constexpr int w = 3;
    static constexpr int r = 0;
    static constexpr int g = 1;
    static constexpr int b = 2;
    static constexpr int a = 3;
    static constexpr int s0 = 0;
    static constexpr int s1 = 1;
    static constexpr int s2 = 2;
    static constexpr int s3 = 3;
    static constexpr int s4 = 4;
    static constexpr int s5 = 5;
    static constexpr int s6 = 6;
    static constexpr int s7 = 7;
    static constexpr int s8 = 8;
    static constexpr int s9 = 9;
    static constexpr int sA = 10;
    static constexpr int sB = 11;
    static constexpr int sC = 12;
    static constexpr int sD = 13;
    static constexpr int sE = 14;
    static constexpr int sF = 15;
};

} // namespace simsycl::sycl

namespace simsycl::detail {

template<typename DataT, int... Indices>
class swizzled_vec;


template<typename DataT, typename VecLike>
struct vec_like_num_elements {};

template<typename DataT, std::convertible_to<DataT> ElementT>
struct vec_like_num_elements<DataT, ElementT> : std::integral_constant<int, 1> {};

template<typename DataT, int... Indices>
struct vec_like_num_elements<DataT, swizzled_vec<DataT, Indices...>> : std::integral_constant<int, sizeof...(Indices)> {
};

template<typename DataT, int N>
struct vec_like_num_elements<DataT, sycl::vec<DataT, N>> : std::integral_constant<int, N> {};

template<typename T, typename DataT>
concept VecLike = vec_like_num_elements<DataT, T>::value > 0;

template<typename T, typename DataT, int NumElements>
concept VecCompatible
    = vec_like_num_elements<DataT, T>::value == 1 || vec_like_num_elements<DataT, T>::value == NumElements;

template<typename From, typename To>
concept explicitly_convertible_to = requires { static_cast<To>(std::declval<From>()); };

template<int... Is>
struct no_repeat_indices;

template<int I1>
struct no_repeat_indices<I1> : std::true_type {};

template<int I1, int... Is>
struct no_repeat_indices<I1, Is...> : std::bool_constant<((I1 != Is) && ...) && no_repeat_indices<Is...>::value> {};

template<int... Is>
static constexpr bool no_repeat_indices_v = no_repeat_indices<Is...>::value;


template<int... Is>
struct index_list;

template<>
struct index_list<> {
    constexpr int operator[](const int /* at */) const { return -1; }
};

template<int I1, int... Is>
struct index_list<I1, Is...> {
    constexpr int operator[](const int at) const { return at == 0 ? I1 : index_list<Is...>{}[at - 1]; }
};


template<typename T>
struct is_vec : std::false_type {};

template<typename DataT, int NumElements>
struct is_vec<sycl::vec<DataT, NumElements>> : std::true_type {};

template<typename T>
static constexpr bool is_vec_v = is_vec<T>::value;


template<typename T>
struct is_swizzle : std::false_type {};

template<typename DataT, int... Indices>
struct is_swizzle<swizzled_vec<DataT, Indices...>> : std::true_type {};

template<typename T>
static constexpr bool is_swizzle_v = is_swizzle<T>::value;


template<typename DataT, int NumElements>
sycl::vec<DataT, NumElements> to_vec(const sycl::vec<DataT, NumElements> &v) {
    return v;
}
template<typename DataT, int... Indices>
sycl::vec<std::remove_const_t<DataT>, sizeof...(Indices)> to_vec(const swizzled_vec<DataT, Indices...> &v) {
    return v;
}
template<typename DataT, int NumElements>
sycl::vec<DataT, NumElements> to_vec(const DataT &e) {
    return sycl::vec<DataT, NumElements>(e);
}


template<typename SwizzledVecWithPartialIndices, int... RemainingIndices>
struct make_swizzled_vec_for_lo_indices;

template<typename DataT, int... SwizzleIndices>
struct make_swizzled_vec_for_lo_indices<swizzled_vec<DataT, SwizzleIndices...>> {
    using type = detail::swizzled_vec<DataT, SwizzleIndices...>;
};

template<typename DataT, int... SwizzleIndices, int NextIndex, int... RemainingIndices>
struct make_swizzled_vec_for_lo_indices<swizzled_vec<DataT, SwizzleIndices...>, NextIndex, RemainingIndices...> {
    using type = std::conditional_t<(sizeof...(SwizzleIndices) >= 1 + sizeof...(RemainingIndices)),
        swizzled_vec<DataT, SwizzleIndices...>,
        typename make_swizzled_vec_for_lo_indices<swizzled_vec<DataT, SwizzleIndices..., NextIndex>,
            RemainingIndices...>::type>;
};

template<typename DataT, int... Indices>
using swizzled_vec_for_lo_indices_t = typename make_swizzled_vec_for_lo_indices<swizzled_vec<DataT>, Indices...>::type;


template<typename SwizzledVecWithPartialIndices, int... DiscardedIndices>
struct make_swizzled_vec_for_hi_indices;

template<typename DataT, int... DiscardedIndices>
struct make_swizzled_vec_for_hi_indices<swizzled_vec<DataT>, DiscardedIndices...> {
    using type = detail::swizzled_vec<DataT>;
};

template<typename DataT, int... RemainingIndices, int NextIndex, int... DiscardedIndices>
struct make_swizzled_vec_for_hi_indices<swizzled_vec<DataT, NextIndex, RemainingIndices...>, DiscardedIndices...> {
    using type = std::conditional_t<(sizeof...(DiscardedIndices) >= 1 + sizeof...(RemainingIndices)),
        swizzled_vec<DataT, NextIndex, RemainingIndices...>,
        typename make_swizzled_vec_for_hi_indices<swizzled_vec<DataT, RemainingIndices...>, DiscardedIndices...,
            NextIndex>::type>;
};

template<typename DataT, int... Indices>
using swizzled_vec_for_hi_indices_t = typename make_swizzled_vec_for_hi_indices<swizzled_vec<DataT, Indices...>>::type;


template<typename SwizzledVecWithPartialIndices, bool IncludeNextIndex, int... RemainingIndices>
struct make_swizzled_vec_for_alternating_indices;

template<typename DataT, int... SwizzleIndices, bool IncludeNextIndex>
struct make_swizzled_vec_for_alternating_indices<swizzled_vec<DataT, SwizzleIndices...>, IncludeNextIndex> {
    using type = swizzled_vec<DataT, SwizzleIndices...>;
};

template<typename DataT, int... SwizzleIndices, bool IncludeNextIndex, int NextIndex, int... RemainingIndices>
struct make_swizzled_vec_for_alternating_indices<swizzled_vec<DataT, SwizzleIndices...>, IncludeNextIndex, NextIndex,
    RemainingIndices...> {
    using partial_type = std::conditional_t<IncludeNextIndex, swizzled_vec<DataT, SwizzleIndices..., NextIndex>,
        swizzled_vec<DataT, SwizzleIndices...>>;
    using type =
        typename make_swizzled_vec_for_alternating_indices<partial_type, !IncludeNextIndex, RemainingIndices...>::type;
};

template<typename DataT, int... Indices>
using swizzled_vec_for_odd_indices_t =
    typename make_swizzled_vec_for_alternating_indices<swizzled_vec<DataT>, false, Indices...>::type;

template<typename DataT, int... Indices>
using swizzled_vec_for_even_indices_t =
    typename make_swizzled_vec_for_alternating_indices<swizzled_vec<DataT>, true, Indices...>::type;


template<template<typename, int...> typename Template, typename DataT, typename IndexSequence>
struct apply_to_index_sequence;

template<template<typename, int...> typename Template, typename DataT, int... Indices>
struct apply_to_index_sequence<Template, DataT, std::integer_sequence<int, Indices...>> {
    using type = Template<DataT, Indices...>;
};

template<template<typename, int...> typename Template, typename DataT, int NumElements>
using apply_to_indices_of_vec_t =
    typename apply_to_index_sequence<Template, DataT, std::make_integer_sequence<int, NumElements>>::type;


template<typename ReferenceDataT, int... Indices>
class swizzled_vec {
    static_assert(!std::is_volatile_v<ReferenceDataT>);
    static_assert(sizeof...(Indices) > 0);

    static constexpr bool allow_assign = !std::is_const_v<ReferenceDataT> && no_repeat_indices_v<Indices...>;
    static constexpr int num_elements = sizeof...(Indices);
    static constexpr index_list<Indices...> indices{};

  public:
    using element_type = std::remove_const_t<ReferenceDataT>;
    using value_type = element_type;

    swizzled_vec() = delete;
    swizzled_vec(const swizzled_vec &) = delete;
    swizzled_vec(swizzled_vec &&) = delete;
    swizzled_vec &operator=(const swizzled_vec &) = delete;
    swizzled_vec &operator=(swizzled_vec &&) = delete;

    template<std::convertible_to<value_type> T>
    swizzled_vec &operator=(const T &rhs)
        requires(allow_assign)
    {
        for(size_t i = 0; i < num_elements; ++i) { m_elems[indices[i]] = rhs; }
        return *this;
    }

    swizzled_vec &operator=(const sycl::vec<value_type, num_elements> &rhs)
        requires(allow_assign)
    {
        for(size_t i = 0; i < num_elements; ++i) { m_elems[indices[i]] = rhs[i]; }
        return *this;
    }

    template<int... OtherIndices>
    swizzled_vec &operator=(const swizzled_vec<value_type, OtherIndices...> &rhs)
        requires(num_elements == sizeof...(OtherIndices) && allow_assign)
    {
        for(size_t i = 0; i < num_elements; ++i) { m_elems[indices[i]] = rhs.m_elems[rhs.indices[i]]; }
        return *this;
    }

    operator sycl::vec<value_type, num_elements>() const {
        return sycl::vec<value_type, num_elements>(m_elems[Indices]...);
    }

    operator value_type() const
        requires(num_elements == 1)
    {
        return m_elems[indices[0]];
    }

    template<detail::explicitly_convertible_to<value_type> T>
    explicit operator T() const
        requires(num_elements == 1)
    {
        return m_elems[indices[0]];
    }

    static constexpr size_t byte_size() noexcept { return sycl::vec<value_type, num_elements>::byte_size(); }

    static constexpr size_t size() noexcept { return num_elements; }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_size() const { return byte_size(); }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_count() const { return size(); }

    template<typename ConvertT, sycl::rounding_mode RoundingMode = sycl::rounding_mode::automatic>
    sycl::vec<ConvertT, num_elements> convert() const {
        return sycl::vec<value_type, num_elements>(*this).template convert<ConvertT, RoundingMode>();
    }

    template<typename AsT>
    AsT as() const {
        return sycl::vec<value_type, num_elements>(*this).template as<AsT>();
    }

    // swizzling

    template<int... SubSwizzleIndices>
        requires((num_elements > SubSwizzleIndices) && ...)
    auto swizzle() const {
        return detail::swizzled_vec<ReferenceDataT, indices[SubSwizzleIndices]...>(m_elems);
    }

#define SIMSYCL_DETAIL_DEFINE_1D_SWIZZLE(req, comp)                                                                    \
    ReferenceDataT &comp() const                                                                                       \
        requires(req && num_elements > sycl::elem::comp)                                                               \
    {                                                                                                                  \
        return m_elems[indices[sycl::elem::comp]];                                                                     \
    }

#define SIMSYCL_DETAIL_DEFINE_2D_SWIZZLE(req, comp1, comp2)                                                            \
    auto comp1##comp2() const                                                                                          \
        requires(req && num_elements > sycl::elem::comp1 && num_elements > sycl::elem::comp2)                          \
    {                                                                                                                  \
        return detail::swizzled_vec<ReferenceDataT, indices[sycl::elem::comp1], indices[sycl::elem::comp2]>(m_elems);  \
    }

#define SIMSYCL_DETAIL_DEFINE_3D_SWIZZLE(req, comp1, comp2, comp3)                                                     \
    auto comp1##comp2##comp3() const                                                                                   \
        requires(req && num_elements > sycl::elem::comp1 && num_elements > sycl::elem::comp2                           \
            && num_elements > sycl::elem::comp3)                                                                       \
    {                                                                                                                  \
        return detail::swizzled_vec<ReferenceDataT, indices[sycl::elem::comp1], indices[sycl::elem::comp2],            \
            indices[sycl::elem::comp3]>(m_elems);                                                                      \
    }

#define SIMSYCL_DETAIL_DEFINE_4D_SWIZZLE(req, comp1, comp2, comp3, comp4)                                              \
    auto comp1##comp2##comp3##comp4() const                                                                            \
        requires(req && num_elements > sycl::elem::comp1 && num_elements > sycl::elem::comp2                           \
            && num_elements > sycl::elem::comp3 && num_elements > sycl::elem::comp4)                                   \
    {                                                                                                                  \
        return detail::swizzled_vec<ReferenceDataT, indices[sycl::elem::comp1], indices[sycl::elem::comp2],            \
            indices[sycl::elem::comp3], indices[sycl::elem::comp4]>(m_elems);                                          \
    }

#include "simsycl/detail/vec_swizzles.inc"

#undef SIMSYCL_DETAIL_DEFINE_4D_SWIZZLE
#undef SIMSYCL_DETAIL_DEFINE_3D_SWIZZLE
#undef SIMSYCL_DETAIL_DEFINE_2D_SWIZZLE
#undef SIMSYCL_DETAIL_DEFINE_1D_SWIZZLE

    auto lo() const
        requires(num_elements > 1)
    {
        return detail::swizzled_vec_for_lo_indices_t<ReferenceDataT, indices[Indices]...>(m_elems);
    }

    auto hi() const
        requires(num_elements > 1)
    {
        return detail::swizzled_vec_for_hi_indices_t<ReferenceDataT, indices[Indices]...>(m_elems);
    }

    auto odd() const
        requires(num_elements > 1)
    {
        return detail::swizzled_vec_for_odd_indices_t<ReferenceDataT, indices[Indices]...>(m_elems);
    }

    auto even() const
        requires(num_elements > 1)
    {
        return detail::swizzled_vec_for_even_indices_t<ReferenceDataT, indices[Indices]...>(m_elems);
    }

    // load and store member functions

    template<sycl::access::address_space AddressSpace, sycl::access::decorated IsDecorated>
    void load(size_t offset, sycl::multi_ptr<const value_type, AddressSpace, IsDecorated> ptr) const
        requires(allow_assign)
    {
        for(int i = 0; i < num_elements; ++i) { m_elems[indices[i]] = ptr[offset + i]; }
    }

    void load(size_t offset, const value_type *ptr) const
        requires(allow_assign)
    {
        for(int i = 0; i < num_elements; ++i) { m_elems[indices[i]] = ptr[offset + i]; }
    }

    template<sycl::access::address_space AddressSpace, sycl::access::decorated IsDecorated>
    void store(size_t offset, sycl::multi_ptr<value_type, AddressSpace, IsDecorated> ptr) const {
        for(int i = 0; i < num_elements; ++i) { ptr[offset + i] = m_elems[indices[i]]; }
    }

    void store(size_t offset, value_type *ptr) const {
        for(int i = 0; i < num_elements; ++i) { ptr[offset + i] = m_elems[indices[i]]; }
    }

    ReferenceDataT &operator[](int index) const {
        SIMSYCL_CHECK(index >= 0 && index < num_elements && "Index out of range");
        return m_elems[indices[index]];
    }

    // operators

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(op, enable_if)                                              \
    template<VecCompatible<element_type, num_elements> LHS, VecCompatible<element_type, num_elements> RHS>             \
    friend constexpr auto operator op(const LHS &lhs, const RHS &rhs)                                                  \
        requires(enable_if                                                                                             \
            && (std::is_same_v<swizzled_vec, LHS> || (std::is_same_v<swizzled_vec, RHS> && !is_swizzle_v<LHS>)))       \
    {                                                                                                                  \
        return to_vec<element_type, num_elements>(lhs) op to_vec<element_type, num_elements>(rhs);                     \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(*, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(/, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(%, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(&, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(|, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(^, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(<<, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(>>, !detail::is_floating_point_v<value_type>)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(op, enable_if)                                           \
    template<VecCompatible<element_type, num_elements> RHS>                                                            \
    friend constexpr swizzled_vec &operator op##=(swizzled_vec && lhs, const RHS & rhs)                                \
        requires(enable_if && allow_assign)                                                                            \
    {                                                                                                                  \
        return lhs = to_vec<element_type, num_elements>(lhs) op to_vec<element_type, num_elements>(rhs);               \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(*, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(/, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(%, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(&, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(|, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(^, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(<<, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(>>, !detail::is_floating_point_v<value_type>)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(op, enable_if)                                               \
    friend constexpr auto operator op(const swizzled_vec &v)                                                           \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        return op to_vec<element_type, num_elements>(v);                                                               \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(~, !detail::is_floating_point_v<value_type>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(!, !detail::is_floating_point_v<value_type>)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR(op)                                                        \
    friend constexpr swizzled_vec &operator op(swizzled_vec && v)                                                      \
        requires(!std::is_same_v<value_type, bool>)                                                                    \
    {                                                                                                                  \
        for(int i = 0; i < num_elements; ++i) { op v.m_elems[indices[i]]; }                                            \
        return v;                                                                                                      \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR(--)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR(op)                                                       \
    friend constexpr auto operator op(swizzled_vec &&v, int)                                                           \
        requires(!std::is_same_v<value_type, bool>)                                                                    \
    {                                                                                                                  \
        auto result = to_vec(v);                                                                                       \
        for(int i = 0; i < num_elements; ++i) { v.m_elems[indices[i]] op; }                                            \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR(--)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(op)                                                          \
    template<VecCompatible<element_type, num_elements> LHS, VecCompatible<element_type, num_elements> RHS>             \
    friend constexpr auto operator op(const LHS &lhs, const RHS &rhs)                                                  \
        requires(std::is_same_v<swizzled_vec, LHS> || (std::is_same_v<swizzled_vec, RHS> && !is_swizzle_v<LHS>))       \
    {                                                                                                                  \
        return to_vec<element_type, num_elements>(lhs) op to_vec<element_type, num_elements>(rhs);                     \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(==)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(!=)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(<)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(<=)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(>=)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(&&)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(||)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR

  private:
    template<typename T, int NumElements>
    friend class sycl::vec;

    template<typename T, int... Is>
    friend class swizzled_vec;

    constexpr swizzled_vec(ReferenceDataT *elems) : m_elems(elems) {}

    ReferenceDataT *m_elems;
};

template<typename DataT, int NumElements>
constexpr size_t vec_alignment_v = std::min(size_t{64}, sizeof(DataT) * NumElements);

template<typename DataT>
constexpr size_t vec_alignment_v<DataT, 3> = std::min(size_t{64}, sizeof(DataT) * 4);

} // namespace simsycl::detail

namespace simsycl::sycl {

template<typename DataT, int NumElements>
class alignas(detail::vec_alignment_v<DataT, NumElements>) vec {
    static_assert(!std::is_const_v<DataT> && !std::is_volatile_v<DataT>);

  private:
    constexpr static int num_elements = NumElements;

  public:
    using element_type = DataT;
    using value_type = DataT;

    using vector_t = vec; // __SYCL_DEVICE_ONLY__

    vec() = default;

    explicit(num_elements > 1) constexpr vec(const DataT &arg) {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = arg; }
    }

    template<detail::VecLike<DataT>... ArgTN>
        requires((detail::vec_like_num_elements<DataT, ArgTN>::value + ...) == NumElements)
    constexpr vec(const ArgTN &...args) {
        init_with_offset<0>(args...);
    }

    vec(const vec &) = default;
    vec &operator=(const vec &rhs) = default;

    template<std::convertible_to<DataT> T>
    vec &operator=(const T &rhs) {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = rhs; }
        return *this;
    }

    // non-standard, but required for disambiguation
    template<int... Indices>
    vec &operator=(const detail::swizzled_vec<DataT, Indices...> &rhs)
        requires(NumElements == sizeof...(Indices))
    {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = rhs.m_elems[rhs.indices[i]]; }
        return *this;
    }

    operator DataT() const
        requires(NumElements == 1)
    {
        return m_elems[0];
    }

    template<detail::explicitly_convertible_to<DataT> T>
    explicit operator T() const
        requires(NumElements == 1)
    {
        return m_elems[0];
    }

    static constexpr size_t byte_size() noexcept { return sizeof m_elems; }

    static constexpr size_t size() noexcept { return NumElements; }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_size() const { return byte_size(); }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_count() const { return size(); }

    template<typename ConvertT, rounding_mode RoundingMode = rounding_mode::automatic>
    vec<ConvertT, NumElements> convert() const {
        static_assert(RoundingMode == rounding_mode::automatic, "other rounding modes not yet implemented");
        vec<ConvertT, NumElements> result;
        for(int i = 0; i < NumElements; ++i) { result[i] = m_elems[i]; }
        return result;
    }

    template<typename AsT>
    AsT as() const {
        static_assert(detail::is_vec_v<AsT>);
        static_assert(sizeof(AsT::m_elems) == sizeof(m_elems));
        AsT reinterpret;
        memcpy(reinterpret.m_elems, m_elems, sizeof(m_elems));
        return reinterpret;
    }

    // swizzling

    template<int... SwizzleIndexes>
        requires((NumElements > SwizzleIndexes) && ...)
    auto swizzle() {
        return detail::swizzled_vec<DataT, SwizzleIndexes...>(m_elems);
    }

    template<int... SwizzleIndexes>
        requires((NumElements > SwizzleIndexes) && ...)
    auto swizzle() const {
        return detail::swizzled_vec<const DataT, SwizzleIndexes...>(m_elems);
    }

#define SIMSYCL_DETAIL_DEFINE_1D_SWIZZLE(req, comp)                                                                    \
    DataT &comp()                                                                                                      \
        requires(req && num_elements > elem::comp)                                                                     \
    {                                                                                                                  \
        return m_elems[elem::comp];                                                                                    \
    }                                                                                                                  \
    const DataT &comp() const                                                                                          \
        requires(req && num_elements > elem::comp)                                                                     \
    {                                                                                                                  \
        return m_elems[elem::comp];                                                                                    \
    }

#define SIMSYCL_DETAIL_DEFINE_2D_SWIZZLE(req, comp1, comp2)                                                            \
    auto comp1##comp2()                                                                                                \
        requires(req && num_elements > elem::comp1 && num_elements > elem::comp2)                                      \
    {                                                                                                                  \
        return detail::swizzled_vec<DataT, elem::comp1, elem::comp2>(m_elems);                                         \
    }                                                                                                                  \
    auto comp1##comp2() const                                                                                          \
        requires(req && num_elements > elem::comp1 && num_elements > elem::comp2)                                      \
    {                                                                                                                  \
        return detail::swizzled_vec<const DataT, elem::comp1, elem::comp2>(m_elems);                                   \
    }

#define SIMSYCL_DETAIL_DEFINE_3D_SWIZZLE(req, comp1, comp2, comp3)                                                     \
    auto comp1##comp2##comp3()                                                                                         \
        requires(req && num_elements > elem::comp1 && num_elements > elem::comp2 && num_elements > elem::comp3)        \
    {                                                                                                                  \
        return detail::swizzled_vec<DataT, elem::comp1, elem::comp2, elem::comp3>(m_elems);                            \
    }                                                                                                                  \
    auto comp1##comp2##comp3() const                                                                                   \
        requires(req && num_elements > elem::comp1 && num_elements > elem::comp2 && num_elements > elem::comp3)        \
    {                                                                                                                  \
        return detail::swizzled_vec<const DataT, elem::comp1, elem::comp2, elem::comp3>(m_elems);                      \
    }

#define SIMSYCL_DETAIL_DEFINE_4D_SWIZZLE(req, comp1, comp2, comp3, comp4)                                              \
    auto comp1##comp2##comp3##comp4()                                                                                  \
        requires(req && num_elements > elem::comp1 && num_elements > elem::comp2 && num_elements > elem::comp3         \
            && num_elements > elem::comp4)                                                                             \
    {                                                                                                                  \
        return detail::swizzled_vec<DataT, elem::comp1, elem::comp2, elem::comp3, elem::comp4>(m_elems);               \
    }                                                                                                                  \
    auto comp1##comp2##comp3##comp4() const                                                                            \
        requires(req && num_elements > elem::comp1 && num_elements > elem::comp2 && num_elements > elem::comp3         \
            && num_elements > elem::comp4)                                                                             \
    {                                                                                                                  \
        return detail::swizzled_vec<const DataT, elem::comp1, elem::comp2, elem::comp3, elem::comp4>(m_elems);         \
    }

#include "simsycl/detail/vec_swizzles.inc"

#undef SIMSYCL_DETAIL_DEFINE_4D_SWIZZLE
#undef SIMSYCL_DETAIL_DEFINE_3D_SWIZZLE
#undef SIMSYCL_DETAIL_DEFINE_2D_SWIZZLE
#undef SIMSYCL_DETAIL_DEFINE_1D_SWIZZLE

    auto lo() const
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_lo_indices_t, const DataT, NumElements>(
            m_elems);
    }

    auto lo()
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_lo_indices_t, DataT, NumElements>(m_elems);
    }

    auto hi() const
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_hi_indices_t, const DataT, NumElements>(
            m_elems);
    }

    auto hi()
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_hi_indices_t, DataT, NumElements>(m_elems);
    }

    auto odd() const
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_odd_indices_t, const DataT, NumElements>(
            m_elems);
    }

    auto odd()
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_odd_indices_t, DataT, NumElements>(m_elems);
    }

    auto even() const
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_even_indices_t, const DataT, NumElements>(
            m_elems);
    }

    auto even()
        requires(num_elements > 1)
    {
        return detail::apply_to_indices_of_vec_t<detail::swizzled_vec_for_even_indices_t, DataT, NumElements>(m_elems);
    }

    // load and store member functions

    template<access::address_space AddressSpace, access::decorated IsDecorated>
    void load(size_t offset, multi_ptr<const DataT, AddressSpace, IsDecorated> ptr) {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = ptr[offset + i]; }
    }

    void load(size_t offset, const DataT *ptr) {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = ptr[offset + i]; }
    }

    template<access::address_space AddressSpace, access::decorated IsDecorated>
    void store(size_t offset, multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const {
        for(int i = 0; i < NumElements; ++i) { ptr[offset + i] = m_elems[i]; }
    }

    void store(size_t offset, DataT *ptr) const {
        for(int i = 0; i < NumElements; ++i) { ptr[offset + i] = m_elems[i]; }
    }

    DataT &operator[](int index) {
        SIMSYCL_CHECK(index >= 0 && index < NumElements && "Index out of range");
        return m_elems[index];
    }

    const DataT &operator[](int index) const {
        SIMSYCL_CHECK(index >= 0 && index < NumElements && "Index out of range");
        return m_elems[index];
    }

    // operators

#define SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(op, enable_if)                                                  \
    friend vec operator op(const vec &lhs, const vec &rhs)                                                             \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        vec result;                                                                                                    \
        for(int i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs.m_elems[i]; }                 \
        return result;                                                                                                 \
    }                                                                                                                  \
    template<std::convertible_to<DataT> T>                                                                             \
    friend vec operator op(const vec &lhs, const T &rhs)                                                               \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        vec result;                                                                                                    \
        for(int i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs; }                            \
        return result;                                                                                                 \
    }                                                                                                                  \
    template<std::convertible_to<DataT> T>                                                                             \
    friend vec operator op(const T &lhs, const vec &rhs)                                                               \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        vec result;                                                                                                    \
        for(int i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs op rhs.m_elems[i]; }                            \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(*, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(/, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(%, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(&, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(|, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(^, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(<<, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR(>>, !detail::is_floating_point_v<DataT>)

#undef SIMSYCL_DETAIL_DEFINE_VEC_BINARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(op, enable_if)                                               \
    friend vec &operator op(vec & lhs, const vec & rhs)                                                                \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        for(int i = 0; i < NumElements; ++i) { lhs.m_elems[i] op rhs.m_elems[i]; }                                     \
        return lhs;                                                                                                    \
    }                                                                                                                  \
    template<int... SwizzleIndices>                                                                                    \
    friend vec &operator op(vec & lhs, const detail::swizzled_vec<DataT, SwizzleIndices...> &rhs)                      \
        requires(enable_if && sizeof...(SwizzleIndices) == NumElements)                                                \
    {                                                                                                                  \
        for(int i = 0; i < NumElements; ++i) { lhs.m_elems[i] op rhs.m_elems[rhs.indices[i]]; }                        \
        return lhs;                                                                                                    \
    }                                                                                                                  \
    template<std::convertible_to<DataT> T>                                                                             \
    friend vec &operator op(vec & lhs, const T & rhs)                                                                  \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        for(int i = 0; i < NumElements; ++i) { lhs.m_elems[i] op rhs; }                                                \
        return lhs;                                                                                                    \
    }

    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(+=, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(-=, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(*=, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(/=, true)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(%=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(&=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(|=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(^=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(<<=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR(>>=, !detail::is_floating_point_v<DataT>)

#undef SIMSYCL_DETAIL_DEFINE_VEC_BINARY_INPLACE_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_VEC_UNARY_COPY_OPERATOR(op, enable_if)                                                   \
    friend constexpr vec operator op(const vec &rhs)                                                                   \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        vec result;                                                                                                    \
        for(int i = 0; i < NumElements; ++i) { result.m_elems[i] = op rhs[i]; }                                        \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_COPY_OPERATOR(~, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_COPY_OPERATOR(!, !detail::is_floating_point_v<DataT>)

#undef SIMSYCL_DETAIL_DEFINE_VEC_UNARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_VEC_UNARY_PREFIX_OPERATOR(op)                                                            \
    friend constexpr vec &operator op(vec & rhs)                                                                       \
        requires(!std::is_same_v<DataT, bool>)                                                                         \
    {                                                                                                                  \
        for(int i = 0; i < NumElements; ++i) { op rhs[i]; }                                                            \
        return rhs;                                                                                                    \
    }

    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_PREFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_PREFIX_OPERATOR(--)

#undef SIMSYCL_DETAIL_DEFINE_VEC_UNARY_PREFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_VEC_UNARY_POSTFIX_OPERATOR(op)                                                           \
    friend constexpr vec operator op(vec &lhs, int)                                                                    \
        requires(!std::is_same_v<DataT, bool>)                                                                         \
    {                                                                                                                  \
        vec result = lhs;                                                                                              \
        for(int i = 0; i < NumElements; ++i) { lhs[i] op; }                                                            \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_POSTFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_POSTFIX_OPERATOR(--)

#undef SIMSYCL_DETAIL_DEFINE_VEC_UNARY_POSTFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(op)                                                              \
    friend vec<decltype(DataT {} op DataT{}), NumElements> operator op(const vec & lhs, const vec & rhs) {             \
        vec<decltype(DataT {} op DataT{}), NumElements> result;                                                        \
        for(int i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs.m_elems[i]; }                 \
        return result;                                                                                                 \
    }                                                                                                                  \
    template<std::convertible_to<DataT> T>                                                                             \
    friend vec<decltype(DataT {} op DataT{}), NumElements> operator op(const vec & lhs, const T & rhs) {               \
        vec<decltype(DataT {} op DataT{}), NumElements> result;                                                        \
        for(int i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs; }                            \
        return result;                                                                                                 \
    }                                                                                                                  \
    template<std::convertible_to<DataT> T>                                                                             \
    friend vec<decltype(DataT {} op DataT{}), NumElements> operator op(const T & lhs, const vec & rhs) {               \
        vec<decltype(DataT {} op DataT{}), NumElements> result;                                                        \
        for(int i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs op rhs.m_elems[i]; }                            \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(==)
    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(!=)
    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(<)
    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(>)
    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(<=)
    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(>=)
    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(&&)
    SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(||)

#undef SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR

  private:
    template<typename T, int... Indices>
    friend class detail::swizzled_vec;
    template<typename T, int N>
    friend class vec;

    template<int Offset, typename... ArgTN>
        requires(Offset + 1 <= NumElements)
    constexpr void init_with_offset(const DataT &arg, const ArgTN &...args) {
        m_elems[Offset] = arg;
        init_with_offset<Offset + 1>(args...);
    }

    template<int Offset, int ArgNumElements, typename... ArgTN>
        requires(Offset + ArgNumElements <= NumElements)
    constexpr void init_with_offset(const vec<DataT, ArgNumElements> &arg, const ArgTN &...args) {
        for(int i = 0; i < ArgNumElements; ++i) { m_elems[Offset + i] = arg.m_elems[i]; }
        init_with_offset<Offset + ArgNumElements>(args...);
    }

    template<int Offset, int... Indices, typename... ArgTN>
        requires(Offset + sizeof...(Indices) <= NumElements)
    constexpr void init_with_offset(const detail::swizzled_vec<DataT, Indices...> &arg, const ArgTN &...args) {
        init_with_offset<Offset>(detail::to_vec(arg), args...);
        init_with_offset<Offset + sizeof...(Indices)>(args...);
    }

    template<int Offset>
        requires(Offset == NumElements)
    constexpr void init_with_offset() {}

    constexpr static int num_storage_elems = NumElements == 3 ? 4 : NumElements;

    // Workaround for friend templates in earlier MSVC versions; TODO: remove when we drop support for them
#if defined(_MSC_VER)
  public:
#endif
    DataT m_elems[num_storage_elems]{};
}; // namespace simsycl::sycl


// Deduction guides
template<class T, class... U>
    requires(std::is_same_v<T, U> && ...)
vec(T, U...) -> vec<T, sizeof...(U) + 1>;


using char2 = vec<int8_t, 2>;
using char3 = vec<int8_t, 3>;
using char4 = vec<int8_t, 4>;
using char8 = vec<int8_t, 8>;
using char16 = vec<int8_t, 16>;

using uchar2 = vec<uint8_t, 2>;
using uchar3 = vec<uint8_t, 3>;
using uchar4 = vec<uint8_t, 4>;
using uchar8 = vec<uint8_t, 8>;
using uchar16 = vec<uint8_t, 16>;

using short2 = vec<int16_t, 2>;
using short3 = vec<int16_t, 3>;
using short4 = vec<int16_t, 4>;
using short8 = vec<int16_t, 8>;
using short16 = vec<int16_t, 16>;

using ushort2 = vec<uint16_t, 2>;
using ushort3 = vec<uint16_t, 3>;
using ushort4 = vec<uint16_t, 4>;
using ushort8 = vec<uint16_t, 8>;
using ushort16 = vec<uint16_t, 16>;

using int2 = vec<int32_t, 2>;
using int3 = vec<int32_t, 3>;
using int4 = vec<int32_t, 4>;
using int8 = vec<int32_t, 8>;
using int16 = vec<int32_t, 16>;

using uint2 = vec<uint32_t, 2>;
using uint3 = vec<uint32_t, 3>;
using uint4 = vec<uint32_t, 4>;
using uint8 = vec<uint32_t, 8>;
using uint16 = vec<uint32_t, 16>;

using long2 = vec<int64_t, 2>;
using long3 = vec<int64_t, 3>;
using long4 = vec<int64_t, 4>;
using long8 = vec<int64_t, 8>;
using long16 = vec<int64_t, 16>;

using ulong2 = vec<uint64_t, 2>;
using ulong3 = vec<uint64_t, 3>;
using ulong4 = vec<uint64_t, 4>;
using ulong8 = vec<uint64_t, 8>;
using ulong16 = vec<uint64_t, 16>;

#if SIMSYCL_FEATURE_HALF_TYPE
using half2 = vec<half, 2>;
using half3 = vec<half, 3>;
using half4 = vec<half, 4>;
using half8 = vec<half, 8>;
using half16 = vec<half, 16>;
#endif

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
using float8 = vec<float, 8>;
using float16 = vec<float, 16>;

using double2 = vec<double, 2>;
using double3 = vec<double, 3>;
using double4 = vec<double, 4>;
using double8 = vec<double, 8>;
using double16 = vec<double, 16>;

} // namespace simsycl::sycl
