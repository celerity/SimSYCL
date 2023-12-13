#pragma once

#include "enums.hh"
#include "forward.hh"
#include "property.hh"
#include "range.hh"

#include <memory>
#include <mutex>

namespace simsycl::sycl::property::image {

class use_host_ptr {
  public:
    use_host_ptr() = default;
};

class use_mutex {
  public:
    use_mutex(std::mutex &mutex_ref);

    std::mutex *get_mutex_ptr() const;
};

class context_bound {
  public:
    context_bound(context bound_context);

    context get_context() const;
};

} // namespace simsycl::sycl::property::image

namespace simsycl::sycl {

template<>
struct is_property<property::image::use_host_ptr> : public std::true_type {};

template<int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_host_ptr, sampled_image<Dimensions, AllocatorT>> : public std::true_type {};

template<int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_host_ptr, unsampled_image<Dimensions, AllocatorT>> : public std::true_type {
};

template<>
struct is_property<property::image::use_mutex> : public std::true_type {};

template<int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_mutex, sampled_image<Dimensions, AllocatorT>> : public std::true_type {};

template<int Dimensions, typename AllocatorT>
struct is_property_of<property::image::use_mutex, unsampled_image<Dimensions, AllocatorT>> : public std::true_type {};

template<>
struct is_property<property::image::context_bound> : public std::true_type {};

template<int Dimensions, typename AllocatorT>
struct is_property_of<property::image::context_bound, sampled_image<Dimensions, AllocatorT>> : public std::true_type {};

template<int Dimensions, typename AllocatorT>
struct is_property_of<property::image::context_bound, unsampled_image<Dimensions, AllocatorT>> : public std::true_type {
};

} // namespace simsycl::sycl

namespace simsycl::sycl {

template<int Dimensions, typename AllocatorT>
class sampled_image : public simsycl::detail::property_interface {
  private:
    using property_compatibility = simsycl::detail::property_compatibility_with<sampled_image<Dimensions, AllocatorT>,
        property::image::use_host_ptr, property::image::use_mutex, property::image::context_bound>;

  public:
    sampled_image(const void *host_pointer, image_format format, image_sampler sampler,
        const range<Dimensions> &range_ref, const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    sampled_image(const void *host_pointer, image_format format, image_sampler sampler,
        const range<Dimensions> &range_ref, const range<Dimensions - 1> &pitch, const property_list &prop_list = {});

    sampled_image(std::shared_ptr<const void> &host_pointer, image_format format, image_sampler sampler,
        const range<Dimensions> &range_ref, const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    sampled_image(std::shared_ptr<const void> &host_pointer, image_format format, image_sampler sampler,
        const range<Dimensions> &range_ref, const range<Dimensions - 1> &pitch, const property_list &prop_list = {});

    /* -- common interface members -- */

    range<Dimensions> get_range() const;

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    range<Dimensions - 1> get_pitch() const;

    size_t byte_size() const;

    size_t size() const;

    template<typename... Ts>
    auto get_access(Ts... args);

    template<typename... Ts>
    auto get_host_access(Ts... args);
};

template<int Dimensions, typename AllocatorT>
class unsampled_image : public simsycl::detail::property_interface {
  private:
    using property_compatibility = simsycl::detail::property_compatibility_with<unsampled_image<Dimensions, AllocatorT>,
        property::image::use_host_ptr, property::image::use_mutex, property::image::context_bound>;

  public:
    unsampled_image(image_format format, const range<Dimensions> &range_ref, const property_list &prop_list = {});

    unsampled_image(image_format format, const range<Dimensions> &range_ref, AllocatorT allocator,
        const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    unsampled_image(image_format format, const range<Dimensions> &range_ref, const range<Dimensions - 1> &pitch,
        const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    unsampled_image(image_format format, const range<Dimensions> &range_ref, const range<Dimensions - 1> &pitch,
        AllocatorT allocator, const property_list &prop_list = {});

    unsampled_image(void *host_pointer, image_format format, const range<Dimensions> &range_ref,
        const property_list &prop_list = {});

    unsampled_image(void *host_pointer, image_format format, const range<Dimensions> &range_ref, AllocatorT allocator,
        const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    unsampled_image(void *host_pointer, image_format format, const range<Dimensions> &range_ref,
        const range<Dimensions - 1> &pitch, const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    unsampled_image(void *host_pointer, image_format format, const range<Dimensions> &range_ref,
        const range<Dimensions - 1> &pitch, AllocatorT allocator, const property_list &prop_list = {});

    unsampled_image(std::shared_ptr<void> &host_pointer, image_format format, const range<Dimensions> &range_ref,
        const property_list &prop_list = {});

    unsampled_image(std::shared_ptr<void> &host_pointer, image_format format, const range<Dimensions> &range_ref,
        AllocatorT allocator, const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    unsampled_image(std::shared_ptr<void> &host_pointer, image_format format, const range<Dimensions> &range_ref,
        const range<Dimensions - 1> &pitch, const property_list &prop_list = {});

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    unsampled_image(std::shared_ptr<void> &host_pointer, image_format format, const range<Dimensions> &range_ref,
        const range<Dimensions - 1> &pitch, AllocatorT allocator, const property_list &prop_list = {});

    /* -- common interface members -- */

    range<Dimensions> get_range() const;

    template<int D = Dimensions, std::enable_if_t<(D > 1), int> = 0>
    range<Dimensions - 1> get_pitch() const;

    size_t byte_size() const noexcept;

    size_t size() const noexcept;

    AllocatorT get_allocator() const;

    template<typename... Ts>
    auto get_access(Ts... args);

    template<typename... Ts>
    auto get_host_access(Ts... args);

    template<typename Destination = std::nullptr_t>
    void set_final_data(Destination final_data = nullptr);

    void set_write_back(bool flag = true);
};

} // namespace simsycl::sycl
