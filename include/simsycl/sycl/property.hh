#pragma once

namespace simsycl::sycl {

template <typename Property>
struct is_property;

template <typename Property>
inline constexpr bool is_property_v = is_property<Property>::value;

template <typename Property, typename SyclObject>
struct is_property_of;

template <typename Property, typename SyclObject>
inline constexpr bool is_property_of_v = is_property_of<Property, SyclObject>::value;

class property_list {
  public:
    template <typename... Properties>
    property_list(Properties... props);
};

} // namespace simsycl::sycl