#pragma once

#include "../detail/check.hh"

#include <algorithm>
#include <any>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace simsycl::detail {

class property_interface;

}

namespace simsycl::sycl {

template <typename Property>
struct is_property : std::false_type {};

template <typename Property>
inline constexpr bool is_property_v = is_property<Property>::value;

template <typename Property, typename SyclObject>
struct is_property_of : std::false_type {};

template <typename Property, typename SyclObject>
inline constexpr bool is_property_of_v = is_property_of<Property, SyclObject>::value;

class property_list {
  public:
    template <typename... Properties>
        requires(is_property_v<Properties> && ...)
    property_list(Properties... props) : m_properties{props...} {}

  private:
    friend class detail::property_interface;

    std::vector<std::any> m_properties;
};

} // namespace simsycl::sycl

namespace simsycl::detail {

template <typename Derived, typename... CompatibleProperties>
struct property_compatibility {};

// TODO must be merged with reference_type
class property_interface {
  public:
    property_interface() = default;

    template <typename Derived, typename... CompatibleProperties>
    explicit property_interface(const sycl::property_list &prop_list,
        property_compatibility<Derived, CompatibleProperties...> /* compatibility */)
        : m_properties(prop_list.m_properties) {
        static_assert((sycl::is_property_v<CompatibleProperties> && ...));
        static_assert((sycl::is_property_of_v<CompatibleProperties, Derived> && ...));
        for(const auto &prop : prop_list.m_properties) {
            SIMSYCL_CHECK(((prop.type() == typeid(CompatibleProperties)) || ...));
        }
    }

    template <typename Property>
    bool has_property() const noexcept {
        return std::any_of(m_properties.begin(), m_properties.end(),
            [](const std::any &prop) { return prop.type() == typeid(Property); });
    }

    template <typename Property>
    Property get_property() const {
        const auto iter = std::find_if(m_properties.begin(), m_properties.end(),
            [](const std::any &prop) { return prop.type() == typeid(Property); });
        SIMSYCL_CHECK(iter != m_properties.end());
        return std::any_cast<Property>(*iter);
    }

  protected:
    const std::vector<std::any> &get_properties() const { return m_properties; }

  private:
    std::vector<std::any> m_properties;
};

} // namespace simsycl::detail
