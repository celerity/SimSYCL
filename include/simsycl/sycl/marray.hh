#pragma once

#include "forward.hh"
#include "type_traits.hh"

#include "../detail/check.hh"

#include <cstddef>
#include <cstdint>

namespace simsycl::detail {

template<typename DataT, typename... ArgTN>
struct marray_init_arg_traits {};

template<typename DataT>
struct marray_init_arg_traits<DataT> {
    static constexpr size_t num_elements = 0;
};

template<typename DataT, std::convertible_to<DataT> ElementT, typename... ArgTN>
struct marray_init_arg_traits<DataT, ElementT, ArgTN...> {
    static constexpr size_t num_elements = 1 + marray_init_arg_traits<DataT, ArgTN...>::num_elements;
};

template<typename DataT, size_t N, typename... ArgTN>
struct marray_init_arg_traits<DataT, sycl::marray<DataT, N>, ArgTN...> {
    static constexpr size_t num_elements = N + marray_init_arg_traits<DataT, ArgTN...>::num_elements;
};

template<typename MArray, typename DataT>
constexpr bool is_marray_v = false;

template<typename DataT, size_t NumElements>
constexpr bool is_marray_v<sycl::marray<DataT, NumElements>, DataT> = true;

} // namespace simsycl::detail

namespace simsycl::sycl {

// SYCL spec says DataT must be a _numeric type as defined by the C++ standard_, but as far as I can tell there is no
// such thing. We could require an arithmetic type instead, but CTS uses a custom type in one test, so we don't for now.
template<typename DataT, size_t NumElements>
class marray {
  public:
    using value_type = DataT;
    using reference = DataT &;
    using const_reference = const DataT &;
    using iterator = DataT *;
    using const_iterator = const DataT *;

    marray() = default;

    explicit constexpr marray(const DataT &arg) {
        for(size_t i = 0; i < NumElements; ++i) { m_elems[i] = arg; }
    }

    template<typename... ArgTN>
        requires(detail::marray_init_arg_traits<DataT, ArgTN...>::num_elements == NumElements)
    constexpr marray(const ArgTN &...args) {
        init_with_offset<0>(args...);
    }

    marray(const marray<DataT, NumElements> &rhs) = default;
    marray(marray<DataT, NumElements> &&rhs) = default;

    marray &operator=(const marray<DataT, NumElements> &rhs) = default;
    marray &operator=(marray<DataT, NumElements> &&rhs) = default;

    marray &operator=(const DataT &rhs) {
        for(int i = 0; i < NumElements; ++i) { m_elems[i] = rhs; }
        return *this;
    }

    operator DataT() const
        requires(NumElements == 1)
    {
        return m_elems[0];
    }

    static constexpr std::size_t size() noexcept { return sizeof(marray); }

    reference operator[](const size_t index) {
        SIMSYCL_CHECK(index < NumElements && "Index out of range");
        return m_elems[index];
    }

    const_reference operator[](const size_t index) const {
        SIMSYCL_CHECK(index < NumElements && "Index out of range");
        return m_elems[index];
    }

    // iterator functions
    iterator begin() { return m_elems; }
    const_iterator begin() const { return m_elems; }

    iterator end() { return m_elems + NumElements; }
    const_iterator end() const { return m_elems + NumElements; }

    // operators

#define SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(op, enable_if)                                               \
    friend marray operator op(const marray &lhs, const marray &rhs)                                                    \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        marray result;                                                                                                 \
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs.m_elems[i]; }              \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend marray operator op(const marray &lhs, const DataT &rhs)                                                     \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        marray result;                                                                                                 \
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs; }                         \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend marray operator op(const DataT &lhs, const marray &rhs)                                                     \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        marray result;                                                                                                 \
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs op rhs.m_elems[i]; }                         \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(*, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(/, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(%, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(&, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(|, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(^, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(<<, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR(>>, !detail::is_floating_point_v<DataT>)

#undef SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(op, enable_if)                                            \
    friend marray &operator op(marray & lhs, const marray & rhs)                                                       \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        for(size_t i = 0; i < NumElements; ++i) { lhs.m_elems[i] op rhs.m_elems[i]; }                                  \
        return lhs;                                                                                                    \
    }                                                                                                                  \
    friend marray &operator op(marray & lhs, const DataT & rhs)                                                        \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        for(size_t i = 0; i < NumElements; ++i) { lhs.m_elems[i] op rhs; }                                             \
        return lhs;                                                                                                    \
    }

    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(+=, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(-=, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(*=, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(/=, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(%=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(&=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(|=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(^=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(<<=, !detail::is_floating_point_v<DataT>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR(>>=, !detail::is_floating_point_v<DataT>)

#undef SIMSYCL_DETAIL_DEFINE_MARRAY_BINARY_INPLACE_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_COPY_OPERATOR(op, enable_if)                                                \
    friend constexpr marray operator op(const marray &rhs)                                                             \
        requires(enable_if)                                                                                            \
    {                                                                                                                  \
        marray result;                                                                                                 \
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = op rhs[i]; }                                     \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_COPY_OPERATOR(+, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_COPY_OPERATOR(-, true)
    SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_COPY_OPERATOR(~, !detail::is_floating_point_v<DataT>)

    friend constexpr marray<bool, NumElements> operator!(const marray &rhs) {
        marray<bool, NumElements> result;
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = !rhs[i]; }
        return result;
    }

#undef SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_PREFIX_OPERATOR(op)                                                         \
    friend constexpr marray &operator op(marray & rhs)                                                                 \
        requires(!std::is_same_v<DataT, bool>)                                                                         \
    {                                                                                                                  \
        for(size_t i = 0; i < NumElements; ++i) { op rhs[i]; }                                                         \
        return rhs;                                                                                                    \
    }

    SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_PREFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_PREFIX_OPERATOR(--)

#undef SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_PREFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_POSTFIX_OPERATOR(op)                                                        \
    friend constexpr marray operator op(marray &lhs, int)                                                              \
        requires(!std::is_same_v<DataT, bool>)                                                                         \
    {                                                                                                                  \
        marray result = lhs;                                                                                           \
        for(size_t i = 0; i < NumElements; ++i) { lhs[i] op; }                                                         \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_POSTFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_POSTFIX_OPERATOR(--)

#undef SIMSYCL_DETAIL_DEFINE_MARRAY_UNARY_POSTFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(op)                                                           \
    friend marray<decltype(DataT {} op DataT{}), NumElements> operator op(const marray & lhs, const marray & rhs) {    \
        marray<decltype(DataT {} op DataT{}), NumElements> result;                                                     \
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs.m_elems[i]; }              \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend marray<decltype(DataT {} op DataT{}), NumElements> operator op(const marray & lhs, const DataT & rhs) {     \
        marray<decltype(DataT {} op DataT{}), NumElements> result;                                                     \
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs.m_elems[i] op rhs; }                         \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend marray<decltype(DataT {} op DataT{}), NumElements> operator op(const DataT & lhs, const marray & rhs) {     \
        marray<decltype(DataT {} op DataT{}), NumElements> result;                                                     \
        for(size_t i = 0; i < NumElements; ++i) { result.m_elems[i] = lhs op rhs.m_elems[i]; }                         \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(==)
    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(!=)
    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(<)
    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(>)
    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(<=)
    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(>=)
    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(&&)
    SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR(||)

#undef SIMSYCL_DETAIL_DEFINE_MARRAY_COMPARISON_OPERATOR

  private:
    template<typename, size_t>
    friend class marray;

    template<size_t Offset, typename... ArgTN>
        requires(Offset + 1 <= NumElements)
    constexpr void init_with_offset(const DataT &arg, const ArgTN &...args) {
        m_elems[Offset] = arg;
        init_with_offset<Offset + 1>(args...);
    }

    template<size_t Offset, size_t ArgNumElements, typename... ArgTN>
        requires(Offset + ArgNumElements <= NumElements)
    constexpr void init_with_offset(const marray<DataT, ArgNumElements> &arg, const ArgTN &...args) {
        for(size_t i = 0; i < ArgNumElements; ++i) { m_elems[Offset + i] = arg.m_elems[i]; }
        init_with_offset<Offset + ArgNumElements>(args...);
    }

    template<size_t Offset>
        requires(Offset == NumElements)
    constexpr void init_with_offset() {}

    // Workaround for friend templates in earlier MSVC versions; TODO: remove when we drop support for them
#if defined(_MSC_VER)
  public:
#endif
    DataT m_elems[NumElements]{};
};

using mchar2 = marray<int8_t, 2>;
using mchar3 = marray<int8_t, 3>;
using mchar4 = marray<int8_t, 4>;
using mchar8 = marray<int8_t, 8>;
using mchar16 = marray<int8_t, 16>;

using muchar2 = marray<uint8_t, 2>;
using muchar3 = marray<uint8_t, 3>;
using muchar4 = marray<uint8_t, 4>;
using muchar8 = marray<uint8_t, 8>;
using muchar16 = marray<uint8_t, 16>;

using mshort2 = marray<int16_t, 2>;
using mshort3 = marray<int16_t, 3>;
using mshort4 = marray<int16_t, 4>;
using mshort8 = marray<int16_t, 8>;
using mshort16 = marray<int16_t, 16>;

using mushort2 = marray<uint16_t, 2>;
using mushort3 = marray<uint16_t, 3>;
using mushort4 = marray<uint16_t, 4>;
using mushort8 = marray<uint16_t, 8>;
using mushort16 = marray<uint16_t, 16>;

using mint2 = marray<int32_t, 2>;
using mint3 = marray<int32_t, 3>;
using mint4 = marray<int32_t, 4>;
using mint8 = marray<int32_t, 8>;
using mint16 = marray<int32_t, 16>;

using muint2 = marray<uint32_t, 2>;
using muint3 = marray<uint32_t, 3>;
using muint4 = marray<uint32_t, 4>;
using muint8 = marray<uint32_t, 8>;
using muint16 = marray<uint32_t, 16>;

using mlong2 = marray<int64_t, 2>;
using mlong3 = marray<int64_t, 3>;
using mlong4 = marray<int64_t, 4>;
using mlong8 = marray<int64_t, 8>;
using mlong16 = marray<int64_t, 16>;

using mulong2 = marray<uint64_t, 2>;
using mulong3 = marray<uint64_t, 3>;
using mulong4 = marray<uint64_t, 4>;
using mulong8 = marray<uint64_t, 8>;
using mulong16 = marray<uint64_t, 16>;

#if SIMSYCL_FEATURE_HALF_TYPE
using mhalf2 = marray<half, 2>;
using mhalf3 = marray<half, 3>;
using mhalf4 = marray<half, 4>;
using mhalf8 = marray<half, 8>;
using mhalf16 = marray<half, 16>;
#endif

using mfloat2 = marray<float, 2>;
using mfloat3 = marray<float, 3>;
using mfloat4 = marray<float, 4>;
using mfloat8 = marray<float, 8>;
using mfloat16 = marray<float, 16>;

using mdouble2 = marray<double, 2>;
using mdouble3 = marray<double, 3>;
using mdouble4 = marray<double, 4>;
using mdouble8 = marray<double, 8>;
using mdouble16 = marray<double, 16>;

using mbool2 = marray<bool, 2>;
using mbool3 = marray<bool, 3>;
using mbool4 = marray<bool, 4>;
using mbool8 = marray<bool, 8>;
using mbool16 = marray<bool, 16>;

} // namespace simsycl::sycl
