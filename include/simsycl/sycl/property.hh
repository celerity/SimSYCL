#pragma once

#include "../detail/check.hh"

#include <algorithm>
#include <any>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace simsycl::detail {

class property_interface;

// outlined into check.cc to avoid cyclic include property.hh -> exception.hh -> context.hh -> property.hh
[[noreturn]] void throw_invalid_property();

} // namespace simsycl::detail

namespace simsycl::sycl {

template<typename Property>
struct is_property : std::false_type {};

template<typename Property>
inline constexpr bool is_property_v = is_property<Property>::value;

template<typename Property, typename SyclObject>
struct is_property_of : std::false_type {};

template<typename Property, typename SyclObject>
inline constexpr bool is_property_of_v = is_property_of<Property, SyclObject>::value;

class property_list {
  public:
    template<typename... Properties>
        requires(is_property_v<Properties> && ...)
    property_list(Properties... props) : m_properties{props...} {}

    // Provided as per 4.5.4.1
    template<typename Property>
    bool has_property() const noexcept {
        return std::any_of(m_properties.begin(), m_properties.end(),
            [](const std::any &prop) { return prop.type() == typeid(Property); });
    }

    // Provided as per 4.5.4.1
    template<typename Property>
    Property get_property() const {
        const auto iter = std::find_if(m_properties.begin(), m_properties.end(),
            [](const std::any &prop) { return prop.type() == typeid(Property); });
        if(iter == m_properties.end()) { detail::throw_invalid_property(); }
        return std::any_cast<Property>(*iter);
    }

  private:
    friend class detail::property_interface;

    std::vector<std::any> m_properties;
};

} // namespace simsycl::sycl

namespace simsycl::detail {

template<typename... CompatibleProperties>
struct property_compatibility {};

template<typename Derived, typename... CompatibleProperties>
struct property_compatibility_with {};

class property_interface {
  public:
    property_interface() = default;

    template<typename... CompatibleProperties>
    explicit property_interface(
        const sycl::property_list &prop_list, property_compatibility<CompatibleProperties...> /* compatibility */)
        : m_properties(prop_list.m_properties) {
        static_assert((sycl::is_property_v<CompatibleProperties> && ...));
        for(const auto &prop : prop_list.m_properties) {
            SIMSYCL_CHECK(((prop.type() == typeid(CompatibleProperties)) || ...));
        }
    }

    template<typename Derived, typename... CompatibleProperties>
    explicit property_interface(const sycl::property_list &prop_list,
        property_compatibility_with<Derived, CompatibleProperties...> /* compatibility */)
        : m_properties(prop_list.m_properties) {
        static_assert((sycl::is_property_v<CompatibleProperties> && ...));
        static_assert((sycl::is_property_of_v<CompatibleProperties, Derived> && ...));
        for(const auto &prop : prop_list.m_properties) {
            SIMSYCL_CHECK(((prop.type() == typeid(CompatibleProperties)) || ...));
        }
    }

    template<typename Property>
    bool has_property() const noexcept {
        return std::any_of(m_properties.begin(), m_properties.end(),
            [](const std::any &prop) { return prop.type() == typeid(Property); });
    }

    template<typename Property>
    Property get_property() const {
        const auto iter = std::find_if(m_properties.begin(), m_properties.end(),
            [](const std::any &prop) { return prop.type() == typeid(Property); });
        if(iter == m_properties.end()) { detail::throw_invalid_property(); }
        return std::any_cast<Property>(*iter);
    }

    friend bool operator==(const property_interface &lhs, const property_interface &rhs) {
        // TODO not correct at all! property_interface should always be used together with reference_type, where
        // equality is shared_ptr identity
        (void)lhs;
        (void)rhs;
        return false;
    }

  protected:
    const std::vector<std::any> &get_properties() const { return m_properties; }

  private:
    std::vector<std::any> m_properties;
};

} // namespace simsycl::detail

namespace simsycl::sycl::property {

struct no_init {};

} // namespace simsycl::sycl::property
