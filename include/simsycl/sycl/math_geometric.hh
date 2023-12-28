#pragma once

#include "math.hh"
#include "vec.hh"

namespace simsycl::detail {

template<typename T>
concept SyclFloat = std::is_same_v<T, float> || std::is_same_v<T, double>
#if SIMSYCL_FEATURE_HALF_TYPE
    || std::is_same_v<T, sycl::half>
#endif
    ;

template<typename T>
concept GeoFloat = //
    SyclFloat<T>
    || ((is_swizzle_v<T> || is_vec_v<T>)&&(num_elements_v<T> > 0 && num_elements_v<T> <= 4)
        && SyclFloat<typename T::element_type>); // TODO: marray

template<typename T>
    requires(is_vec_v<T> || is_swizzle_v<T>)
auto sum(const T &f) {
    auto ret = f[0];
    for(int i = 1; i < num_elements_v<T>; ++i) { ret += f[i]; }
    return ret;
}
template<SyclFloat T>
auto sum(const T &f) {
    return f;
}

template<GeoFloat T>
struct element_type {
    using type = T;
};
template<GeoFloat T>
    requires(is_vec_v<T> || is_swizzle_v<T>)
struct element_type<T> {
    using type = typename T::element_type;
};
template<GeoFloat T>
using element_type_t = typename element_type<T>::type;

template<typename VT, typename T>
auto to_matching_vec(const T &v) {
    return detail::to_vec<detail::element_type_t<VT>, detail::num_elements_v<VT>>(v);
}

} // namespace simsycl::detail

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
