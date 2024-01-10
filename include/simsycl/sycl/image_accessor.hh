#pragma once

#include "enums.hh"
#include "forward.hh"
#include "property.hh"

namespace simsycl::sycl {

template<typename DataT, int Dimensions, access_mode AccessMode, image_target AccessTarget>
class is_property_of<sycl::property::no_init, unsampled_image_accessor<DataT, Dimensions, AccessMode, AccessTarget>>
    : public std::true_type {};

template<typename DataT, int Dimensions, access_mode AccessMode>
class is_property_of<sycl::property::no_init, host_unsampled_image_accessor<DataT, Dimensions, AccessMode>>
    : public std::true_type {};

} // namespace simsycl::sycl
