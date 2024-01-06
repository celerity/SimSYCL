#pragma once

#include <simsycl/sycl/vec.hh>
#include <simsycl/system.hh>

#include <tuple>

namespace simsycl::test {

template<typename DeviceSetup>
void configure_device_with(DeviceSetup &&setup_device) {
    auto system = builtin_system;
    setup_device(system.devices.at("GPU"));
    configure_system(system);
}

template<int Dimensions>
inline bool check_bool_vec(simsycl::sycl::vec<bool, Dimensions> a) {
    for(int i = 0; i < Dimensions; ++i) {
        if(!a[i]) { return false; }
    }
    return true;
}

// From https://stackoverflow.com/a/70405002/1522056
template<typename T1, typename T2>
class tuple_cross_product {
    template<typename T, typename... Ts>
    static auto inner_helper(T &&, std::tuple<Ts...> &&)
        -> decltype(std::make_tuple(std::make_tuple(std::declval<T>(), std::declval<Ts>())...));

    template<typename... Ts, typename T>
    static auto outer_helper(std::tuple<Ts...> &&, T &&)
        -> decltype(std::tuple_cat(inner_helper(std::declval<Ts>(), std::declval<T>())...));

  public:
    using type = decltype(outer_helper(std::declval<T1>(), std::declval<T2>()));
};

}; // namespace simsycl::test
