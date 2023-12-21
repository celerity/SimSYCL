#pragma once

#include "enums.hh"
#include "forward.hh"
#include "type_traits.hh"

#include "../detail/check.hh"
#include "../detail/utils.hh"

#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace simsycl::detail {

template<typename DataT, int NumElements>
constexpr size_t vec_alignment_v = std::min(size_t{64}, sizeof(DataT) * NumElements);

template<typename DataT>
constexpr size_t vec_alignment_v<DataT, 3> = std::min(size_t{64}, sizeof(DataT) * 4);

template<int... Is>
struct no_repeat_indices {
    static_assert(sizeof...(Is) < 0, "Only implemented for up to 4 indices");
};

template<int I1>
struct no_repeat_indices<I1> : std::true_type {};
template<int... Is>
static constexpr bool no_repeat_indices_v = no_repeat_indices<Is...>::value;

template<int I1, int I2>
struct no_repeat_indices<I1, I2> : std::bool_constant<I1 != I2> {};

template<int I1, int I2, int I3>
struct no_repeat_indices<I1, I2, I3> : std::bool_constant<I1 != I2 && I1 != I3 && I2 != I3> {};

template<int I1, int I2, int I3, int I4>
struct no_repeat_indices<I1, I2, I3, I4>
    : std::bool_constant<I1 != I2 && I1 != I3 && I1 != I4 && I2 != I3 && I2 != I4 && I3 != I4> {};

template<typename V1, typename V2>
static constexpr bool is_compatible_vector_v = requires {
    V1::dimensions == V2::dimensions;
    std::is_same_v<typename V1::element_type, typename V2::element_type>;
} || std::is_same_v<typename V1::element_type, V2>;

template<typename DataT, int... Indices>
class swizzled_vec;

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
sycl::vec<DataT, sizeof...(Indices)> to_vec(const swizzled_vec<DataT, Indices...> &v) {
    return v;
}
template<typename DataT, int NumElements>
sycl::vec<DataT, NumElements> to_vec(const DataT &e) {
    return sycl::vec<DataT, NumElements>(e);
}

template<typename DataT, int... Indices>
class swizzled_vec {
    static_assert(!std::is_volatile_v<DataT>);
    static_assert(sizeof...(Indices) > 0);

    static constexpr bool allow_assign = no_repeat_indices_v<Indices...>;

  public:
    using element_type = DataT;
    static constexpr int dimensions = sizeof...(Indices);

    swizzled_vec() = delete;
    swizzled_vec(const swizzled_vec &) = delete;
    swizzled_vec(swizzled_vec &&) = delete;
    swizzled_vec &operator=(const swizzled_vec &) = delete;
    swizzled_vec &operator=(swizzled_vec &&) = delete;

    swizzled_vec &operator=(const DataT &rhs)
        requires(allow_assign)
    {
        for(size_t i = 0; i < dimensions; ++i) { m_elems[indices[i]] = rhs; }
        return *this;
    }

    swizzled_vec &operator=(const sycl::vec<DataT, dimensions> &rhs)
        requires(allow_assign)
    {
        for(size_t i = 0; i < dimensions; ++i) { m_elems[indices[i]] = rhs[i]; }
        return *this;
    }

    template<int... OtherIndices>
    swizzled_vec &operator=(const swizzled_vec<DataT, OtherIndices...> &rhs)
        requires(dimensions == sizeof...(OtherIndices) && allow_assign)
    {
        for(size_t i = 0; i < dimensions; ++i) { m_elems[indices[i]] = rhs.m_elems[rhs.indices[i]]; }
        return *this;
    }

    operator sycl::vec<DataT, dimensions>() const { return sycl::vec<DataT, dimensions>(m_elems[Indices]...); }

    operator DataT() const
        requires(dimensions == 1)
    {
        return m_elems[0];
    }

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(op, enable_if)                                              \
    template<typename LHS, typename RHS>                                                                               \
    friend constexpr auto operator op(const LHS &lhs, const RHS &rhs)                                                  \
        requires(enable_if && is_compatible_vector_v<swizzled_vec, RHS> && is_compatible_vector_v<swizzled_vec, LHS>   \
            && (std::is_same_v<swizzled_vec, LHS> || (std::is_same_v<swizzled_vec, RHS> && !is_swizzle_v<LHS>)))       \
    {                                                                                                                  \
        return to_vec<element_type, dimensions>(lhs) op to_vec<element_type, dimensions>(rhs);                         \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(*, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(/, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(%, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(&, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(|, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(^, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(<<, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR(>>, !detail::is_floating_point_v<DataT>)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(op, enable_if)                                           \
    template<typename RHS>                                                                                             \
    friend constexpr swizzled_vec &operator op##=(swizzled_vec && lhs, const RHS & rhs)                                \
        requires(enable_if && allow_assign && is_compatible_vector_v<swizzled_vec, RHS>)                               \
    {                                                                                                                  \
        return lhs = to_vec<element_type, dimensions>(lhs) op to_vec<element_type, dimensions>(rhs);                   \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(*, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(/, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(%, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(&, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(|, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(^, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(<<, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR(>>, !detail::is_floating_point_v<DataT>)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_BINARY_INPLACE_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(op, enable_if)                                               \
    friend constexpr auto operator op(const swizzled_vec &v)                                                           \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        return op to_vec(v);                                                                                           \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(~, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR(!, !detail::is_floating_point_v<DataT>)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR(op)                                                        \
    friend constexpr swizzled_vec &operator op(swizzled_vec && v)                                                      \
        requires(!std::is_same_v<DataT, bool>)                                                                         \
    {                                                                                                                  \
        for(int d = 0; d < dimensions; ++d) { op v.m_elems[indices[d]]; }                                              \
        return v;                                                                                                      \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR(--)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_PREFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR(op)                                                       \
    friend constexpr auto operator op(swizzled_vec &&v, int)                                                           \
        requires(!std::is_same_v<DataT, bool>)                                                                         \
    {                                                                                                                  \
        auto result = to_vec(v);                                                                                       \
        for(int d = 0; d < dimensions; ++d) { v.m_elems[indices[d]] op; }                                              \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR(--)
#undef SIMSYCL_DETAIL_DEFINE_SWIZZLE_UNARY_POSTFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_SWIZZLE_COMPARISON_OPERATOR(op)                                                          \
    template<typename LHS, typename RHS>                                                                               \
    friend constexpr auto operator op(const LHS &lhs, const RHS &rhs)                                                  \
        requires(is_compatible_vector_v<swizzled_vec, RHS> && is_compatible_vector_v<swizzled_vec, LHS>                \
            && (std::is_same_v<swizzled_vec, LHS> || (std::is_same_v<swizzled_vec, RHS> && !is_swizzle_v<LHS>)))       \
    {                                                                                                                  \
        return to_vec<element_type, dimensions>(lhs) op to_vec<element_type, dimensions>(rhs);                         \
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

    inline static constexpr int indices[] = {Indices...};

    template<int NumElements>
        requires(!std::is_const_v<DataT> && NumElements > detail::max(Indices...))
    swizzled_vec(sycl::vec<DataT, NumElements> &vec) : m_elems(vec.m_elems) {}

    template<int NumElements>
        requires(std::is_const_v<DataT> && NumElements > detail::max(Indices...))
    swizzled_vec(const sycl::vec<std::remove_const_t<DataT>, NumElements> &vec) : m_elems(vec.m_elems) {}

    DataT *m_elems;
};

} // namespace simsycl::detail

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

template<typename DataT, int NumElements>
class alignas(detail::vec_alignment_v<DataT, NumElements>) vec {
    static_assert(!std::is_const_v<DataT> && !std::is_volatile_v<DataT>);

  public:
    using element_type = DataT;
    using value_type = DataT;
    static constexpr int dimensions = NumElements;

    using vector_t = vec; // __SYCL_DEVICE_ONLY__

    vec() = default;

    explicit constexpr vec(const DataT &arg) {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = arg; }
    }

    template<typename... ArgTN>
    constexpr vec(const ArgTN &...args) {
        init_with_offset<0>(args...);
    }

    vec(const vec &) = default;
    vec &operator=(const vec &rhs) = default;

    vec &operator=(const DataT &rhs) {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = rhs; }
        return *this;
    }

    operator DataT() const
        requires(NumElements == 1)
    {
        return m_elems[0];
    }

    static constexpr size_t byte_size() noexcept { return sizeof m_elems; }

    static constexpr size_t size() noexcept { return NumElements; }

    [[deprecated]] size_t get_size() const { return byte_size(); }

    [[deprecated]] size_t get_count() const { return size(); }

    template<typename ConvertT, rounding_mode RoundingMode = rounding_mode::automatic>
    vec<ConvertT, NumElements> convert() const {
        static_assert(RoundingMode == rounding_mode::automatic, "other rounding modes not yet implemented");
        vec<ConvertT, NumElements> result;
        for(int i = 0; i < NumElements; ++i) { result[i] = m_elems[i]; }
        return result;
    }

    template<typename AsT>
    AsT as() const;

    // swizzling

    template<int... SwizzleIndexes>
        requires(NumElements > detail::max(SwizzleIndexes...))
    auto swizzle() {
        return detail::swizzled_vec<DataT, SwizzleIndexes...>(*this);
    }

    template<int... SwizzleIndexes>
        requires(NumElements > detail::max(SwizzleIndexes...))
    auto swizzle() const {
        return detail::swizzled_vec<const DataT, SwizzleIndexes...>(*this);
    }

#define SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(component, enable_if)                                                 \
    auto component()                                                                                                   \
        requires(NumElements > elem::component && enable_if)                                                           \
    {                                                                                                                  \
        return swizzle<elem::component>();                                                                             \
    }                                                                                                                  \
    auto component() const                                                                                             \
        requires(NumElements > elem::component && enable_if)                                                           \
    {                                                                                                                  \
        return swizzle<elem::component>();                                                                             \
    }

    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(x, NumElements <= 4)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(y, NumElements <= 4)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(z, NumElements <= 4)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(w, NumElements <= 4)

    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(r, NumElements == 4)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(g, NumElements == 4)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(b, NumElements == 4)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(a, NumElements == 4)

    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s0, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s1, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s2, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s3, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s4, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s5, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s6, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s7, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s8, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(s9, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(sA, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(sB, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(sC, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(sD, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(sE, true)
    SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE(sF, true)

#undef SIMSYCL_DETAIL_VEC_DEFINE_SCALAR_SWIZZLE

#ifdef SYCL_SIMPLE_SWIZZLES

#define SIMSYCL_SWIZZLE_2(req, comp1, comp2)                                                                           \
    auto comp1##comp2()                                                                                                \
        requires(req && NumElements > elem::comp1 && NumElements > elem::comp2)                                        \
    {                                                                                                                  \
        return detail::swizzled_vec<DataT, elem::comp1, elem::comp2>(*this);                                           \
    }                                                                                                                  \
    auto comp1##comp2() const                                                                                          \
        requires(req && NumElements > elem::comp1 && NumElements > elem::comp2)                                        \
    {                                                                                                                  \
        return detail::swizzled_vec<const DataT, elem::comp1, elem::comp2>(*this);                                     \
    }

#define SIMSYCL_SWIZZLE_3(req, comp1, comp2, comp3)                                                                    \
    auto comp1##comp2##comp3()                                                                                         \
        requires(req && NumElements > elem::comp1 && NumElements > elem::comp2 && NumElements > elem::comp3)           \
    {                                                                                                                  \
        return detail::swizzled_vec<DataT, elem::comp1, elem::comp2, elem::comp3>(*this);                              \
    }                                                                                                                  \
    auto comp1##comp2##comp3() const                                                                                   \
        requires(req && NumElements > elem::comp1 && NumElements > elem::comp2 && NumElements > elem::comp3)           \
    {                                                                                                                  \
        return detail::swizzled_vec<const DataT, elem::comp1, elem::comp2, elem::comp3>(*this);                        \
    }

#define SIMSYCL_SWIZZLE_4(req, comp1, comp2, comp3, comp4)                                                             \
    auto comp1##comp2##comp3##comp4()                                                                                  \
        requires(req && NumElements > elem::comp1 && NumElements > elem::comp2 && NumElements > elem::comp3            \
            && NumElements > elem::comp4)                                                                              \
    {                                                                                                                  \
        return detail::swizzled_vec<DataT, elem::comp1, elem::comp2, elem::comp3, elem::comp4>(*this);                 \
    }                                                                                                                  \
    auto comp1##comp2##comp3() const                                                                                   \
        requires(req && NumElements > elem::comp1 && NumElements > elem::comp2 && NumElements > elem::comp3            \
            && NumElements > elem::comp4)                                                                              \
    {                                                                                                                  \
        return detail::swizzled_vec<const DataT, elem::comp1, elem::comp2, elem::comp3, elem::comp4>(*this);           \
    }

#include "simsycl/detail/vec_swizzles.inc"

#undef SIMSYCL_SWIZZLE_2
#undef SIMSYCL_SWIZZLE_3
#undef SIMSYCL_SWIZZLE_4

#endif // #ifdef SYCL_SIMPLE_SWIZZLES

    auto lo() const
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<const DataT, 0>(*this);
    }
    auto lo()
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<DataT, 0>(*this);
    }
    auto lo() const
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<const DataT, 0, 1>(*this);
    }
    auto lo()
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<DataT, 0, 1>(*this);
    }

    auto hi() const
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<const DataT, 1>(*this);
    }
    auto hi()
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<DataT, 1>(*this);
    }
    auto hi() const
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<const DataT, 2, 3>(*this);
    }
    auto hi()
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<DataT, 2, 3>(*this);
    }

    auto odd() const
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<const DataT, 1>(*this);
    }
    auto odd()
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<DataT, 1>(*this);
    }
    auto odd() const
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<const DataT, 1, 3>(*this);
    }
    auto odd()
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<DataT, 1, 3>(*this);
    }

    auto even() const
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<const DataT, 0>(*this);
    }
    auto even()
        requires(NumElements == 2)
    {
        return detail::swizzled_vec<DataT, 0>(*this);
    }
    auto even() const
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<const DataT, 0, 2>(*this);
    }
    auto even()
        requires(NumElements >= 3 && NumElements <= 4)
    {
        return detail::swizzled_vec<DataT, 0, 2>(*this);
    }

    // load and store member functions

    template<access::address_space AddressSpace, access::decorated IsDecorated>
    void load(size_t offset, multi_ptr<const DataT, AddressSpace, IsDecorated> ptr);

    template<access::address_space AddressSpace, access::decorated IsDecorated>
    void store(size_t offset, multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const;

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
        for(int d = 0; d < NumElements; ++d) { result.m_elems[d] = lhs.m_elems[d] op rhs.m_elems[d]; }                 \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend vec operator op(const vec &lhs, const DataT &rhs)                                                           \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        vec result;                                                                                                    \
        for(int d = 0; d < NumElements; ++d) { result.m_elems[d] = lhs.m_elems[d] op rhs; }                            \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend vec operator op(const DataT &lhs, const vec &rhs)                                                           \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        vec result;                                                                                                    \
        for(int d = 0; d < NumElements; ++d) { result.m_elems[d] = lhs op rhs.m_elems[d]; }                            \
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
        for(int d = 0; d < NumElements; ++d) { lhs.m_elems[d] op rhs.m_elems[d]; }                                     \
        return lhs;                                                                                                    \
    }                                                                                                                  \
    friend vec &operator op(vec & lhs, const DataT & rhs)                                                              \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        for(int d = 0; d < NumElements; ++d) { lhs.m_elems[d] op rhs; }                                                \
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
        for(int d = 0; d < NumElements; ++d) { result.m_elems[d] = op rhs[d]; }                                        \
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
        for(int d = 0; d < NumElements; ++d) { op rhs[d]; }                                                            \
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
        for(int d = 0; d < NumElements; ++d) { lhs[d] op; }                                                            \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_POSTFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_VEC_UNARY_POSTFIX_OPERATOR(--)

#undef SIMSYCL_DETAIL_DEFINE_VEC_UNARY_POSTFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_VEC_COMPARISON_OPERATOR(op)                                                              \
    friend vec<decltype(DataT {} op DataT{}), NumElements> operator op(const vec & lhs, const vec & rhs) {             \
        vec<decltype(DataT {} op DataT{}), NumElements> result;                                                        \
        for(int d = 0; d < NumElements; ++d) { result.m_elems[d] = lhs.m_elems[d] op rhs.m_elems[d]; }                 \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend vec<decltype(DataT {} op DataT{}), NumElements> operator op(const vec & lhs, const DataT & rhs) {           \
        vec<decltype(DataT {} op DataT{}), NumElements> result;                                                        \
        for(int d = 0; d < NumElements; ++d) { result.m_elems[d] = lhs.m_elems[d] op rhs; }                            \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend vec<decltype(DataT {} op DataT{}), NumElements> operator op(const DataT & lhs, const vec & rhs) {           \
        vec<decltype(DataT {} op DataT{}), NumElements> result;                                                        \
        for(int d = 0; d < NumElements; ++d) { result.m_elems[d] = lhs op rhs.m_elems[d]; }                            \
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
    template<typename T, int Dimensions>
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
        for(int i = 0; i < ArgNumElements; ++i) { m_elems[Offset + i] = arg[i]; }
        m_elems[Offset] = arg;
        init_with_offset<Offset + ArgNumElements>(args...);
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
