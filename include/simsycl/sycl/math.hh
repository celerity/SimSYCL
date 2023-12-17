#pragma once

#include "vec.hh"

#include <cmath>


#define SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(name)                                             \
    template<typename T, int N>                                                                                        \
    vec<T, N> name(const vec<T, N> &x) {                                                                               \
        vec<T, N> result;                                                                                              \
        for(int i = 0; i < N; ++i) { result[i] = name(x[i]); }                                                         \
        return result;                                                                                                 \
    }

#define SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(name)                                            \
    template<typename T, int N>                                                                                        \
    vec<T, N> name(const vec<T, N> &x, const vec<T, N> &y) {                                                           \
        vec<T, N> result;                                                                                              \
        for(int i = 0; i < N; ++i) { result[i] = name(x[i], y[i]); }                                                   \
        return result;                                                                                                 \
    }


namespace simsycl::sycl {

using std::acos;
using std::acosh;
// TODO acospi
using std::asin;
// TODO asinpi
using std::atan;
using std::atan2;
using std::atanh;
// TODO atanpi
// TODO atan2pi
using std::cbrt;
using std::ceil;
using std::copysign;
using std::cos;
using std::cosh;
// TODO cospi
using std::erf;
using std::erfc;
using std::exp;
using std::exp2;
// TODO exp10
using std::fabs;
using std::fdim;
using std::floor;
using std::fma;
using std::fmax;
using std::fmin;
using std::fmod;
// TODO fract
using std::frexp;
using std::hypot;
using std::ilogb;
using std::ldexp;
using std::lgamma;
// TODO lgamma_r
using std::log;
using std::log10;
using std::log1p;
using std::log2;
using std::logb;
// TODO mad
// TODO maxmag
// TODO minmag
using std::modf;
// TODO nan (different signature from std::nan)
using std::nextafter;
using std::pow;
// TODO pown
// TODO powr
using std::remainder;
using std::remquo;
using std::rint;
// TODO rootn
using std::round;
// TODO rsqrt
using std::sin;
// TODO sincos
using std::sinh;
// TODO sinpi
using std::sqrt;
using std::tan;
using std::tanh;
// TODO tanpi
using std::tgamma;
using std::trunc;

SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(acos)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(acosh)
// TODO acospi
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(asin)
// TODO asinpi
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(atan)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(atan2)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(atanh)
// TODO atanpi
// TODO atan2pi
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(cbrt)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(ceil)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(copysign)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(cos)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(cosh)
// TODO cospi
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(erf)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(erfc)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(exp)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(exp2)
// TODO exp10
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(fabs)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(fdim)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(floor)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(fma)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(fmax)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(fmin)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(fmod)
// TODO fract
// TODO frexp
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(hypot)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(ilogb)
// TODO ldexp
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(lgamma)
// TODO lgamma_r
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(log)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(log10)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(log1p)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(log2)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(logb)
// TODO mad
// TODO maxmag
// TODO minmag
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(modf)
// TODO nan (different signature from nan)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(nextafter)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(pow)
// TODO pown
// TODO powr
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(remainder)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(remquo)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(rint)
// TODO rootn
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(round)
// TODO rsqrt
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(sin)
// TODO sincos
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(sinh)
// TODO sinpi
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(sqrt)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(tan)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(tanh)
// TODO tanpi
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(tgamma)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(trunc)

// TODO integer math

} // namespace simsycl::sycl

namespace simsycl::sycl::native {

using sycl::cos;
// TODO divide
using sycl::exp;
using sycl::exp2;
// TODO exp10
using sycl::log;
using sycl::log10;
using sycl::log2;
// TODO powr
// TODO recip
// TODO rsqrt
using sycl::sin;
using sycl::sqrt;
using sycl::tan;

} // namespace simsycl::sycl::native

namespace simsycl::sycl::half_precision {

using sycl::cos;
// TODO divide
using sycl::exp;
using sycl::exp2;
// TODO exp10
using sycl::log;
using sycl::log10;
using sycl::log2;
// TODO powr
// TODO recip
// TODO rsqrt
using sycl::sin;
using sycl::sqrt;
using sycl::tan;

} // namespace simsycl::sycl::half_precision

namespace simsycl::sycl {

using std::abs;
// TODO abs_diff
// TODO add_sat
// TODO hadd
// TODO rhadd
using std::clamp;
// TODO clz
// TODO ctz
// TODO mad_hi
// TODO mad_sat
using std::max;
using std::min;
// TODO mul_hi
// TODO rotate
// TODO sub_sat
// TODO upsample
// TODO popcount
// TODO mad24
// TODO mul24

SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(abs)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(clamp)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(max)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(min)

} // namespace simsycl::sycl

#undef SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION
#undef SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION
