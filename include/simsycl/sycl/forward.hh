#pragma once

#include "allocator.hh"
#include "enums.hh"

#include <type_traits>

namespace simsycl::sycl {

template <typename DataT, int Dimensions = 1,
    access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write),
    target AccessTarget = target::device, access::placeholder IsPlaceholder = access::placeholder::false_t>
class accessor;

template <typename T, access::address_space AddressSpace = access::address_space::global_space>
class atomic;

template <typename T, int Dimensions = 1, typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
class buffer;

class context;

class device;

class event;

class exception;

class exception_list;

class handler;

struct half; // TODO

template <typename DataT, int Dimensions>
class host_sampled_image_accessor;

template <typename DataT, int Dimensions = 1,
    access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write)>
class host_unsampled_image_accessor;

template <int Dimensions = 1>
class id;

template <int Dimensions = 1, bool WithOffset = true>
class item;

template <int Dimensions = 1>
class nd_item;

class kernel;

template <bundle_state State>
class kernel_bundle;

template <typename DataT, int Dimensions = 1>
class local_accessor;

template <typename ElementType, access::address_space Space, access::decorated DecorateAddress>
class multi_ptr;

template <int Dimensions = 1>
class nd_range;

template <int Dimensions = 1>
class group;

class sub_group;

class platform;

class queue;

template <int Dimensions = 1>
class range;

template <int Dimensions = 1, typename AllocatorT = image_allocator>
class sampled_image;

template <typename DataT, int Dimensions, image_target AccessTarget = image_target::device>
class sampled_image_accessor;

template <typename DataT, int Dimensions, access_mode AccessMode, image_target AccessTarget = image_target::device>
class unsampled_image_accessor;

template <int Dimensions = 1, typename AllocatorT = image_allocator>
class unsampled_image;

} // namespace simsycl::sycl

namespace simsycl::detail {

class unnamed_kernel;

struct nd_item_impl;
struct group_impl;
struct sub_group_impl;

sycl::sub_group make_sub_group(
    const sycl::id<1> &, const sycl::range<1> &, const sycl::id<1> &, const sycl::range<1> &, sub_group_impl *);

sub_group_impl &get_group_impl(const sycl::sub_group &g);
template <int Dimensions>
group_impl &get_group_impl(const sycl::group<Dimensions> &g);

template <typename T, int Dimensions, typename AllocatorT>
T *get_buffer_data(sycl::buffer<T, Dimensions, AllocatorT> &buf);

sycl::handler make_handler();

void **require_local_memory(sycl::handler &cgh, size_t size, size_t align);

} // namespace simsycl::detail
