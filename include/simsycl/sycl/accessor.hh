#pragma once

#include "enums.hh"
#include "forward.hh"
#include "id.hh"
#include "property.hh"
#include "range.hh"

#include "../detail/subscript.hh"

#include <iterator>

namespace simsycl::detail {

template <simsycl::sycl::access_mode AccessMode, simsycl::sycl::target Target>
struct accessor_tag {
    inline static constexpr simsycl::sycl::access_mode access_mode = AccessMode;
    inline static constexpr simsycl::sycl::target target = Target;
};

template <typename DataT, int Dimensions>
struct nd_iterator {
    // TODO
};

} // namespace simsycl::detail

namespace simsycl::sycl::property {

struct no_init {};

} // namespace simsycl::sycl::property

namespace simsycl::sycl {

template <>
struct is_property<property::no_init> : std::true_type {};

inline constexpr simsycl::detail::accessor_tag<access_mode::read, target::device> read_only;
inline constexpr simsycl::detail::accessor_tag<access_mode::write, target::device> write_only;
inline constexpr simsycl::detail::accessor_tag<access_mode::read_write, target::device> read_write;
inline constexpr simsycl::detail::accessor_tag<access_mode::read, target::host_task> read_only_host_task;
inline constexpr simsycl::detail::accessor_tag<access_mode::write, target::host_task> write_only_host_task;
inline constexpr simsycl::detail::accessor_tag<access_mode::read_write, target::host_task> read_write_host_task;

inline constexpr property::no_init no_init;

template <typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget,
    access::placeholder IsPlaceholder>
class accessor {
  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    template <access::decorated IsDecorated>
    using accessor_ptr = std::enable_if_t<AccessTarget == target::device,
        multi_ptr<value_type, access::address_space::global_space, IsDecorated>>;

    using iterator = simsycl::detail::nd_iterator<value_type, Dimensions>;
    using const_iterator = simsycl::detail::nd_iterator<const value_type, Dimensions>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    accessor();

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, const property_list &prop_list = {});

    template <typename AllocatorT, typename TagT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, TagT tag, const property_list &prop_list = {});

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        const property_list &prop_list = {});

    template <typename AllocatorT, typename TagT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref, TagT tag,
        const property_list &prop_list = {});

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        const property_list &prop_list = {});

    template <typename AllocatorT, typename TagT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range, TagT tag,
        const property_list &prop_list = {});

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, const property_list &prop_list = {});

    template <typename AllocatorT, typename TagT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, TagT tag, const property_list &prop_list = {});

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, const property_list &prop_list = {});

    template <typename AllocatorT, typename TagT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, TagT tag, const property_list &prop_list = {});

    template <typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, id<Dimensions> access_offset, const property_list &prop_list = {});

    template <typename AllocatorT, typename TagT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, id<Dimensions> access_offset, TagT tag, const property_list &prop_list = {});

    /* -- common interface members -- */

    void swap(accessor &other);

    bool is_placeholder() const;

    size_type byte_size() const noexcept;

    size_type size() const noexcept;

    size_type max_size() const noexcept;

    [[deprecated]] size_t get_size() const;

    [[deprecated]] size_t get_count() const;

    bool empty() const noexcept;

    range<Dimensions> get_range() const;

    id<Dimensions> get_offset() const;

    template <access_mode A = AccessMode>
        requires(A != access_mode::atomic)
    reference operator[](id<Dimensions> index) const;

    template <access_mode A = AccessMode>
        requires(A == access_mode::atomic)
    [[deprecated]] atomic<DataT, access::address_space::global_space> operator[](id<Dimensions> index) const;

    decltype(auto) operator[](size_t index) const { return detail::subscript(*this, index); }

    std::add_pointer_t<value_type> get_pointer() const noexcept;

    template <access::decorated IsDecorated>
    accessor_ptr<IsDecorated> get_multi_ptr() const noexcept;

    iterator begin() const noexcept;

    iterator end() const noexcept;

    const_iterator cbegin() const noexcept;

    const_iterator cend() const noexcept;

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;
};

template <typename DataT, access_mode AccessMode, target AccessTarget, access::placeholder IsPlaceholder>
class accessor<DataT, 0, AccessMode, AccessTarget, IsPlaceholder> {
  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    template <access::decorated IsDecorated>
    using accessor_ptr = std::enable_if_t<AccessTarget == target::device,
        multi_ptr<value_type, access::address_space::global_space, IsDecorated>>;

    using iterator = simsycl::detail::nd_iterator<value_type, 0>;
    using const_iterator = simsycl::detail::nd_iterator<const value_type, 0>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    accessor();

    template <typename AllocatorT>
    accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, const property_list &prop_list = {});

    template <typename AllocatorT>
    accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        const property_list &prop_list = {});

    /* -- common interface members -- */

    void swap(accessor &other);

    bool is_placeholder() const;

    size_type byte_size() const noexcept;

    size_type size() const noexcept;

    size_type max_size() const noexcept;

    [[deprecated]] size_t get_size() const;

    [[deprecated]] size_t get_count() const;

    bool empty() const noexcept;

    template <access_mode A = AccessMode, std::enable_if_t<A != access_mode::atomic, int> = 0>
    operator reference() const;

    template <access_mode A = AccessMode, std::enable_if_t<A != access_mode::atomic && A != access_mode::read, int> = 0>
    const accessor &operator=(const value_type &other) const;

    template <access_mode A = AccessMode, std::enable_if_t<A != access_mode::atomic && A != access_mode::read, int> = 0>
    const accessor &operator=(value_type &&other) const;

    template <access_mode A = AccessMode, std::enable_if_t<A == access_mode::atomic, int> = 0>
    [[deprecated]] operator atomic<DataT, access::address_space::global_space>() const;

    std::add_pointer_t<value_type> get_pointer() const noexcept;

    template <access::decorated IsDecorated>
    accessor_ptr<IsDecorated> get_multi_ptr() const noexcept;

    iterator begin() const noexcept;

    iterator end() const noexcept;

    const_iterator cbegin() const noexcept;

    const_iterator cend() const noexcept;

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;
};

} // namespace simsycl::sycl
