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

#if SIMSYCL_FEATURE_HALF_TYPE

#define SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(name)                                                       \
    inline half name(const half x) { return static_cast<half>(name(static_cast<float>(x))); }

#define SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(name)                                                      \
    inline half name(const half x, const half y) {                                                                     \
        return static_cast<half>(name(static_cast<float>(x), static_cast<float>(y)));                                  \
    }

#define SIMSYCL_DETAIL_MATH_DERIVE_TERNARY_FUNCTION_FOR_HALF(name)                                                     \
    inline half name(const half x, const half y, const half z) {                                                       \
        return static_cast<half>(name(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));           \
    }

#else

#define SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(name)
#define SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(name)
#define SIMSYCL_DETAIL_MATH_DERIVE_TERNARY_FUNCTION_FOR_HALF(name)

#endif

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

SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(acos);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(acosh);
// TODO acospi
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(asin);
// TODO asinpi
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(atan);
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(atan2)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(atanh)
// TODO atanpi
// TODO atan2pi
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(cbrt);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(ceil);
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(copysign)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(cos);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(cosh);
// TODO cospi
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(erf);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(erfc);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(exp)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(exp2);
// TODO exp10
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(fabs);
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(fdim);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(floor);
SIMSYCL_DETAIL_MATH_DERIVE_TERNARY_FUNCTION_FOR_HALF(fma)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(fmax)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(fmin)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(fmod)
// TODO fract
// TODO frexp - has an int argument
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(hypot)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(ilogb)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(ldexp)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(lgamma);
// TODO lgamma_r
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(log)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(log10);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(log1p);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(log2);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(logb);
// TODO mad
// TODO maxmag
// TODO minmag
// TODO modf - has an int* second parameter
// TODO nan (different signature from std::nan)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(nextafter);
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(pow)
// TODO pown
// TODO powr
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(remainder)
// TODO remquo - has a float* second parameter
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(rint);
// TODO root)n
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(round);
// TODO rsqrt
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(sin);
// TODO sincos
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(sinh);
// TODO sinpi
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(sqrt);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(tan);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(tanh);
// TODO tanpi
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(tgamma);
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(trunc);

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
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(fdim)
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

namespace simsycl::sycl {

// TODO isequal
// TODO isnotequal
using std::isfinite;
using std::isgreater;
using std::isgreaterequal;
using std::isinf;
using std::isless;
using std::islessequal;
using std::islessgreater;
using std::isnan;
using std::isnormal;
// TODO isordered
// TODO isunordered
using std::signbit;
// TODO any
// TODO all
// TODO bitselect
// TODO select

SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(isgreater)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(isgreaterequal)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(isless)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(islessequal)
SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF(islessgreater)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(isfinite)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(isinf)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(isnan)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(isnormal)
SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF(signbit)

SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(isgreater)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(isgreaterequal)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(isless)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(islessequal)
SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION(islessgreater)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(isfinite)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(isinf)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(isnan)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(isnormal)
SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION(signbit)

} // namespace simsycl::sycl

#undef SIMSYCL_DETAIL_MATH_DEFINE_UNARY_COMPONENT_WISE_VEC_FUNCTION
#undef SIMSYCL_DETAIL_MATH_DEFINE_BINARY_COMPONENT_WISE_VEC_FUNCTION
#undef SIMSYCL_DETAIL_MATH_DERIVE_UNARY_FUNCTION_FOR_HALF
#undef SIMSYCL_DETAIL_MATH_DERIVE_BINARY_FUNCTION_FOR_HALF
#undef SIMSYCL_DETAIL_MATH_DERIVE_TERNARY_FUNCTION_FOR_HALF
