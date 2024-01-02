#pragma once

#include "../detail/math_utils.hh"

namespace simsycl::sycl {

template<detail::SyclScalar T1>
auto clamp(T1 x, T1 minval, T1 maxval) {
    return std::clamp(x, minval, maxval);
}

template<detail::NonScalar T1, detail::NonScalar T2, detail::NonScalar T3>
auto clamp(T1 x, T2 minval, T3 maxval) {
    return detail::component_wise_op(x, minval, maxval,
        [](T1::value_type e, T2::value_type min, T3::value_type max) { return std::clamp(e, min, max); });
}

template<detail::NonScalar T1>
auto clamp(T1 x, typename T1::value_type minval, typename T1::value_type maxval) {
    return simsycl::sycl::clamp(x, detail::to_matching_vec<T1>(minval), detail::to_matching_vec<T1>(maxval));
}


} // namespace simsycl::sycl
