#pragma once

#include "../detail/math_utils.hh"

namespace simsycl::sycl {

// Note: arguments are passed by const ref rather than value to avoid gcc warnings
//       the standard requires pass-by-value, but I'm not sure if this is visible to the user

// TODO cross

template<detail::GeoFloat T1, detail::GeoFloat T2>
auto dot(const T1 &f, const T2 &g) {
    return detail::sum(detail::to_matching_vec<T1>(f) * detail::to_matching_vec<T2>(g));
}

template<detail::GeoFloat T>
auto length(const T &f) {
    return sqrt(detail::sum(pow(detail::to_matching_vec<T>(f), detail::to_matching_vec<T>(2))));
}

template<detail::GeoFloat T1, detail::GeoFloat T2>
auto distance(const T1 &p0, const T2 &p1) {
    return length(p1 - p0);
}

template<detail::GeoFloat T>
auto normalize(const T &f) {
    return detail::to_matching_vec<T>(f) / detail::to_matching_vec<T>(length(f));
}

// TODO fast_distance
// TODO fast_length
// TODO fast_normalize

} // namespace simsycl::sycl
