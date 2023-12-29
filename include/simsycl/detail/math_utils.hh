#pragma once

#include "../sycl/marray.hh"
#include "../sycl/vec.hh"

namespace simsycl::detail {

template<typename T>
concept SyclFloat = std::is_same_v<T, float> || std::is_same_v<T, double>
#if SIMSYCL_FEATURE_HALF_TYPE
    || std::is_same_v<T, sycl::half>
#endif
    ;

template<SyclFloat T>
struct num_elements<T> : std::integral_constant<int, 1> {};

template<typename T>
concept GenFloat
    = SyclFloat<T> || ((is_swizzle_v<T> || is_vec_v<T> || is_marray_v<T>)&&SyclFloat<typename T::element_type>);

template<typename T>
concept GeoFloat = SyclFloat<T>
    || ((is_swizzle_v<T> || is_vec_v<T> || is_marray_v<T>)&&(num_elements_v<T> > 0 && num_elements_v<T> <= 4)
        && SyclFloat<typename T::value_type>);

template<typename T>
    requires(is_vec_v<T> || is_swizzle_v<T> || is_marray_v<T>)
auto sum(const T &f) {
    auto ret = f[0];
    for(int i = 1; i < num_elements_v<T>; ++i) { ret += f[i]; }
    return ret;
}
template<SyclFloat T>
auto sum(const T &f) {
    return f;
}

template<typename T>
struct element_type {
    using type = T;
};
template<typename T>
    requires(is_vec_v<T> || is_swizzle_v<T>)
struct element_type<T> {
    using type = typename T::element_type;
};
template<typename T>
    requires(is_marray_v<T>)
struct element_type<T> {
    using type = typename T::value_type;
};
template<typename T>
using element_type_t = typename element_type<T>::type;

template<typename DataT, int NumElements>
sycl::vec<DataT, NumElements> marray_to_vec(const sycl::marray<DataT, NumElements> &v) {
    sycl::vec<DataT, NumElements> ret;
    for(int i = 0; i < NumElements; ++i) { ret[i] = v[i]; }
    return ret;
}

template<typename VT, typename T>
    requires(!is_marray_v<T>)
sycl::vec<element_type_t<VT>, num_elements_v<VT>> to_matching_vec(const T &v) {
    return to_vec<element_type_t<VT>, num_elements_v<VT>>(v);
}
template<typename VT, typename T>
    requires(is_marray_v<T>)
sycl::vec<element_type_t<VT>, num_elements_v<VT>> to_matching_vec(const T &v) {
    return marray_to_vec<element_type_t<VT>, num_elements_v<VT>>(v);
}

} // namespace simsycl::detail
