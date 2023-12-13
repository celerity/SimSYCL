#pragma once

#include "check.hh"

#include <cstdlib>
#include <functional>
#include <type_traits>

namespace simsycl::detail {

template<typename Interface, int Dimensions>
class coordinate {
  public:
    constexpr static int dimensions = Dimensions;

    coordinate() = default;

    template<typename... Values,
        typename
        = std::enable_if_t<sizeof...(Values) + 1 == Dimensions && (... && std::is_convertible_v<Values, size_t>)>>
    constexpr coordinate(const size_t dim_0, const Values... dim_n) : m_values{dim_0, static_cast<size_t>(dim_n)...} {}

    constexpr size_t get(int dimension) {
        SIMSYCL_CHECK(dimension < Dimensions);
        return m_values[dimension];
    }

    constexpr size_t &operator[](int dimension) {
        SIMSYCL_CHECK(dimension < Dimensions);
        return m_values[dimension];
    }

    constexpr size_t operator[](int dimension) const {
        SIMSYCL_CHECK(dimension < Dimensions);
        return m_values[dimension];
    }

    friend constexpr bool operator==(const Interface &lhs, const Interface &rhs) {
        bool equal = true;
        for(int d = 0; d < Dimensions; ++d) { equal &= lhs[d] == rhs[d]; }
        return equal;
    }

    friend constexpr bool operator!=(const Interface &lhs, const Interface &rhs) { return !(lhs == rhs); }

#define SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(op)                                                  \
    friend constexpr Interface operator op(const Interface &lhs, const Interface &rhs) {                               \
        Interface result = make_interface_type();                                                                      \
        for(int d = 0; d < Dimensions; ++d) { result[d] = lhs.m_values[d] op rhs.m_values[d]; }                        \
        return result;                                                                                                 \
    }                                                                                                                  \
    friend constexpr Interface operator op(const Interface &lhs, const size_t &rhs) {                                  \
        Interface result = make_interface_type();                                                                      \
        for(int d = 0; d < Dimensions; ++d) { result[d] = lhs.m_values[d] op rhs; }                                    \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(+)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(-)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(*)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(/)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(%)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<<)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>>)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(&)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(|)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(^)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(&&)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(||)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>=)

#undef SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(op)                                               \
    friend constexpr Interface &operator op(Interface &lhs, const Interface &rhs) {                                    \
        for(int d = 0; d < Dimensions; ++d) { lhs.m_values[d] op rhs.m_values[d]; }                                    \
        return lhs;                                                                                                    \
    }                                                                                                                  \
    friend constexpr Interface &operator op(Interface &lhs, const size_t &rhs) {                                       \
        for(int d = 0; d < Dimensions; ++d) { lhs.m_values[d] op rhs; }                                                \
        return lhs;                                                                                                    \
    }

    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(+=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(-=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(*=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(/=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(%=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(<<=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(>>=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(&=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(|=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(^=)

#undef SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(op)                                                  \
    friend constexpr Interface operator op(const size_t &lhs, const Interface &rhs) {                                  \
        Interface result = make_interface_type();                                                                      \
        for(int d = 0; d < Dimensions; ++d) { result[d] = lhs op rhs.m_values[d]; }                                    \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(+)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(-)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(*)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(/)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(%)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<<)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>>)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(&)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(|)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(^)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(&&)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(||)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<=)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>=)

#undef SIMSYCL_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(op)                                                       \
    friend constexpr Interface operator op(const Interface &rhs) {                                                     \
        Interface result = make_interface_type();                                                                      \
        for(int d = 0; d < Dimensions; ++d) { result[d] = op rhs[d]; }                                                 \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(+)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(-)

#undef SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(op)                                                     \
    friend constexpr Interface &operator op(Interface &rhs) {                                                          \
        for(int d = 0; d < Dimensions; ++d) { op rhs[d]; }                                                             \
        return rhs;                                                                                                    \
    }

    SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(--)

#undef SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR

#define SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(op)                                                    \
    friend constexpr Interface operator op(Interface &lhs, int) {                                                      \
        Interface result = lhs = make_interface_type();                                                                \
        for(int d = 0; d < Dimensions; ++d) { lhs[d] op; }                                                             \
        return result;                                                                                                 \
    }

    SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(++)
    SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(--)

#undef SIMSYCL_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR

  private:
    size_t m_values[Dimensions];

    // interface type construction helper to use in friend operators
    // (because friendship is not transitive)
    static Interface make_interface_type() { return {}; }
};

} // namespace simsycl::detail
