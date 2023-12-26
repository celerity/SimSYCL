#pragma once

#include "allocator.hh"
#include "enums.hh"

#include "../detail/preprocessor.hh"
#include <simsycl/config.hh>

#include <cstdint>
#include <functional>
#include <type_traits>


namespace boost::context {
class continuation;
}

namespace simsycl {

struct platform_config;
struct device_config;
struct system_config;

} // namespace simsycl

namespace simsycl::sycl {

// access::placeholder
SIMSYCL_START_IGNORING_DEPRECATIONS

template<typename DataT, int Dimensions = 1,
    access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write),
    target AccessTarget = target::device, access::placeholder IsPlaceholder = access::placeholder::false_t>
class accessor;

SIMSYCL_STOP_IGNORING_DEPRECATIONS

template<typename T, access::address_space AddressSpace = access::address_space::global_space>
class atomic;

template<typename T, int Dimensions = 1, typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
class buffer;

using byte [[deprecated]] = std::uint8_t;

class context;

class device;

class event;

class exception;

class exception_list;

class handler;

#if SIMSYCL_FEATURE_HALF_TYPE
using half = _Float16; // currently requires a compiler that supports _Float16
#endif

template<typename DataT, int Dimensions = 1,
    access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write)>
class host_accessor;

template<typename DataT, int Dimensions>
class host_sampled_image_accessor;

template<typename DataT, int Dimensions = 1,
    access_mode AccessMode = (std::is_const_v<DataT> ? access_mode::read : access_mode::read_write)>
class host_unsampled_image_accessor;

template<int Dimensions = 1>
class id;

class interop_handle;

template<int Dimensions = 1, bool WithOffset = true>
class item;

template<int Dimensions = 1>
class nd_item;

class kernel;

template<bundle_state State>
class kernel_bundle;

class kernel_id;

class kernel_handler;

template<typename DataT, int Dimensions = 1>
class local_accessor;

template<typename DataT, size_t NumElements>
class marray;

template<typename ElementType, access::address_space Space,
    access::decorated DecorateAddress = access::decorated::legacy>
class multi_ptr;

template<int Dimensions = 1>
class nd_range;

template<int Dimensions = 1>
class group;

class sub_group;

class platform;

class property_list;

class queue;

template<int Dimensions = 1>
class range;

template<int Dimensions = 1, typename AllocatorT = image_allocator>
class sampled_image;

template<typename DataT, int Dimensions, image_target AccessTarget = image_target::device>
class sampled_image_accessor;

template<typename DataT, int Dimensions, access_mode AccessMode, image_target AccessTarget = image_target::device>
class unsampled_image_accessor;

template<int Dimensions = 1, typename AllocatorT = image_allocator>
class unsampled_image;

template<typename DataT, int NumElements>
class vec;

} // namespace simsycl::sycl

namespace simsycl::detail {

class unnamed_kernel;

struct concurrent_nd_item;
struct concurrent_group;
struct concurrent_sub_group;

using device_selector = std::function<int(const sycl::device &)>;

sycl::sub_group make_sub_group(
    const sycl::id<1> &, const sycl::range<1> &, const sycl::id<1> &, const sycl::range<1> &, concurrent_sub_group *);

concurrent_sub_group &get_concurrent_group(const sycl::sub_group &g);
template<int Dimensions>
concurrent_group &get_concurrent_group(const sycl::group<Dimensions> &g);

template<typename T, int Dimensions, typename AllocatorT>
T *get_buffer_data(sycl::buffer<T, Dimensions, AllocatorT> &buf);

sycl::handler make_handler(const sycl::device &device);

sycl::interop_handle make_interop_handle();

void **require_local_memory(sycl::handler &cgh, size_t size, size_t align);

struct event_state;

sycl::event make_event(std::shared_ptr<event_state> &&state);

void enter_kernel_fiber(boost::context::continuation &&from_scheduler);
boost::context::continuation &&leave_kernel_fiber();
void yield_to_kernel_scheduler();
void maybe_yield_to_kernel_scheduler();

} // namespace simsycl::detail
