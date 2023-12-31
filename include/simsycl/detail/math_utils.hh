#pragma once

#include "../sycl/marray.hh"
#include "../sycl/vec.hh"

namespace simsycl::detail {

struct undefined_num_elements {};

#define SIMSYCL_DETAIL_DEFINE_NUM_ELEMENTS_TRAIT(trait_name, concept_name, include_scalar)                             \
    template<typename T>                                                                                               \
    struct trait_name : std::conditional_t<concept_name<T> && include_scalar, std::integral_constant<int, 1>,          \
                            undefined_num_elements> {};                                                                \
                                                                                                                       \
    template<concept_name DataT, int... Indices>                                                                       \
    struct trait_name<swizzled_vec<DataT, Indices...>> : std::integral_constant<int, sizeof...(Indices)> {};           \
                                                                                                                       \
    template<concept_name DataT, int NumElements>                                                                      \
    struct trait_name<sycl::vec<DataT, NumElements>> : std::integral_constant<int, NumElements> {};                    \
                                                                                                                       \
    template<concept_name DataT, size_t NumElements>                                                                   \
    struct trait_name<sycl::marray<DataT, NumElements>> : std::integral_constant<int, static_cast<int>(NumElements)> { \
    };                                                                                                                 \
                                                                                                                       \
    template<typename T>                                                                                               \
    constexpr int trait_name##_v = trait_name<T>::value;


template<typename T>
concept SyclFloat = std::is_same_v<T, float> || std::is_same_v<T, double>
#if SIMSYCL_FEATURE_HALF_TYPE
    || std::is_same_v<T, sycl::half>
#endif
    ;

template<typename T>
concept SyclInt = std::is_same_v<T, char> || std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>
    || std::is_same_v<T, short> || std::is_same_v<T, unsigned short> || std::is_same_v<T, int>
    || std::is_same_v<T, unsigned int> || std::is_same_v<T, long> || std::is_same_v<T, unsigned long>
    || std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>;

template<typename T>
concept SyclScalar = SyclFloat<T> || SyclInt<T>;

SIMSYCL_DETAIL_DEFINE_NUM_ELEMENTS_TRAIT(gen_float_num_elements, SyclFloat, true /* include scalar */)
SIMSYCL_DETAIL_DEFINE_NUM_ELEMENTS_TRAIT(gen_int_num_elements, SyclInt, true /* include scalar */)
SIMSYCL_DETAIL_DEFINE_NUM_ELEMENTS_TRAIT(non_scalar_float_num_elements, SyclFloat, false /* include scalar */)
SIMSYCL_DETAIL_DEFINE_NUM_ELEMENTS_TRAIT(non_scalar_int_num_elements, SyclInt, false /* include scalar */)

template<typename T>
concept GenFloat = gen_float_num_elements<T>::value >= 0;

template<typename T>
concept GeoFloat = gen_float_num_elements<T>::value > 0 && gen_float_num_elements<T>::value <= 4;

template<typename T>
concept GenInt = gen_int_num_elements<T>::value >= 0;

template<typename T>
concept Generic = GenFloat<T> || GenInt<T>;

template<typename T>
struct generic_num_elements : std::conditional_t<GenFloat<T>, gen_float_num_elements<T>, gen_int_num_elements<T>> {};

template<typename T>
constexpr int generic_num_elements_v = generic_num_elements<T>::value;

template<typename T>
concept NonScalarFloat = non_scalar_float_num_elements<T>::value >= 0;

template<typename T>
concept NonScalarInt = non_scalar_int_num_elements<T>::value >= 0;

template<typename T>
concept NonScalar = NonScalarFloat<T> || NonScalarInt<T>;

template<typename T>
struct non_scalar_num_elements
    : std::conditional_t<NonScalarFloat<T>, non_scalar_float_num_elements<T>, non_scalar_int_num_elements<T>> {};

template<typename T>
constexpr int non_scalar_num_elements_v = non_scalar_num_elements<T>::value;


template<NonScalar T>
auto sum(const T &f) {
    auto ret = f[0];
    for(int i = 1; i < non_scalar_num_elements_v<T>; ++i) { ret += f[i]; }
    return ret;
}

template<SyclScalar T>
auto sum(const T &f) {
    return f;
}

template<typename T>
struct element_type {};

template<SyclScalar T>
struct element_type<T> {
    using type = T;
};

template<NonScalar T>
struct element_type<T> {
    using type = typename T::value_type;
};

template<typename T>
using element_type_t = typename element_type<T>::type;

template<typename DataT, size_t NumElements>
sycl::vec<DataT, NumElements> marray_to_vec(const sycl::marray<DataT, NumElements> &v) {
    sycl::vec<DataT, NumElements> ret;
    for(size_t i = 0; i < NumElements; ++i) { ret[i] = v[i]; }
    return ret;
}

template<Generic VT, typename T>
sycl::vec<element_type_t<VT>, generic_num_elements_v<VT>> to_matching_vec(const T &v) {
    if constexpr(is_marray_v<T>) {
        return marray_to_vec(v);
    } else {
        return to_vec<element_type_t<VT>, generic_num_elements_v<VT>>(v);
    }
}

} // namespace simsycl::detail
