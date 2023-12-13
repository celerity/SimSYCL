#pragma once

#include <cmath>

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

} // namespace simsycl::sycl
