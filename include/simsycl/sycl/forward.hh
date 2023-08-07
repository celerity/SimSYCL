#pragma once

#include "enums.hh"

#include <type_traits>

namespace simsycl::sycl {

template <typename DataT, int Dimensions = 1,
    access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write),
    target AccessTarget = target::device, access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor;

class context;

class device;

class event;

class exception;

class exception_list;

class handler;

template <int Dimensions = 1>
class id;

template <int Dimensions = 1, bool WithOffset = true>
class item;

class kernel;

template <bundle_state State>
class kernel_bundle;

template <int Dimensions = 1>
class nd_range;

template <int Dimensions = 1>
class range;

} // namespace simsycl::sycl
