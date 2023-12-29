#pragma once

#include "../detail/math_utils.hh"

namespace simsycl::sycl {

// Note: arguments are passed by const ref rather than value to avoid gcc warnings
//       the standard requires pass-by-value, but I'm not sure if this is visible to the user

// TODO cross
// TODO dot

template<detail::GeoFloat T>
auto length(const T &f) {
    return sqrt(detail::sum(pow(detail::to_matching_vec<T>(f), detail::to_matching_vec<T>(2))));
}

template<detail::GeoFloat T1, detail::GeoFloat T2>
auto distance(const T1 &p0, const T2 &p1) {
    return length(p1 - p0);
}

// TODO normalize
// TODO fast_distance
// TODO fast_length
// TODO fast_normalize

} // namespace simsycl::sycl
