#pragma once

#include "buffer.hh"
#include "enums.hh"
#include "forward.hh"
#include "id.hh"
#include "multi_ptr.hh"
#include "property.hh"
#include "range.hh"

#include "../detail/subscript.hh"
#include "../detail/utils.hh"

#include <iterator>
#include <limits>

// access::placeholder, access_mode::atomic
SIMSYCL_START_IGNORING_DEPRECATIONS

namespace simsycl::detail {

template<simsycl::sycl::access_mode AccessMode, simsycl::sycl::target Target>
struct accessor_tag {
    inline static constexpr simsycl::sycl::access_mode access_mode = AccessMode;
    inline static constexpr simsycl::sycl::target target = Target;
};

template<typename Accessor, typename DataT, int Dimensions>
class accessor_iterator {
  public:
    using value_type = DataT;
    using difference_type = std::make_signed_t<size_t>;
    using reference = value_type &;
    using pointer = value_type *;
    using iterator_category = std::random_access_iterator_tag;

    accessor_iterator() = default;

    reference operator*() const {
        SIMSYCL_CHECK(m_accessor != nullptr);
        if constexpr(Dimensions > 0) {
            SIMSYCL_CHECK(m_linear_index < m_accessor->get_range().size());
            return (*m_accessor)[m_accessor->get_offset()
                + detail::linear_index_to_id(m_accessor->get_range(), static_cast<size_t>(m_linear_index))];
        } else {
            SIMSYCL_CHECK(m_linear_index == 0);
            return *m_accessor;
        }
    }

    reference operator[](const difference_type n) const { return *(*this + n); }
    pointer operator->() const { return &**this; }

    friend accessor_iterator &operator+=(accessor_iterator &lhs, const difference_type n) { return lhs.advance(n); }
    friend accessor_iterator &operator-=(accessor_iterator &lhs, const difference_type n) { return lhs.advance(-n); }

    friend accessor_iterator operator+(accessor_iterator lhs, const difference_type n) { return lhs.advance(n); }
    friend accessor_iterator operator-(accessor_iterator lhs, const difference_type n) { return lhs.advance(-n); }

    friend difference_type operator-(const accessor_iterator &lhs, const accessor_iterator &rhs) {
        SIMSYCL_CHECK(lhs.m_accessor == rhs.m_accessor);
        return lhs.m_linear_index - rhs.m_linear_index;
    }

    friend accessor_iterator &operator++(accessor_iterator &lhs) { return lhs.advance(+1); }
    friend accessor_iterator &operator--(accessor_iterator &lhs) { return lhs.advance(-1); }

    friend accessor_iterator operator++(accessor_iterator &lhs, int) {
        auto before = lhs;
        lhs.advance(+1);
        return before;
    }

    friend accessor_iterator operator--(accessor_iterator &lhs, int) {
        auto before = lhs;
        lhs.advance(-1);
        return before;
    }

    friend bool operator==(const accessor_iterator &lhs, const accessor_iterator &rhs) {
        return lhs.m_linear_index == rhs.m_linear_index && lhs.m_accessor == rhs.m_accessor;
    }

    friend bool operator!=(const accessor_iterator &lhs, const accessor_iterator &rhs) { return !(lhs == rhs); }

    friend bool operator<(const accessor_iterator &lhs, const accessor_iterator &rhs) {
        SIMSYCL_CHECK(lhs.m_accessor == rhs.m_accessor);
        return lhs.m_linear_index < rhs.m_linear_index;
    }

    friend bool operator<=(const accessor_iterator &lhs, const accessor_iterator &rhs) {
        SIMSYCL_CHECK(lhs.m_accessor == rhs.m_accessor);
        return lhs.m_linear_index <= rhs.m_linear_index;
    }

    friend bool operator>(const accessor_iterator &lhs, const accessor_iterator &rhs) {
        SIMSYCL_CHECK(lhs.m_accessor == rhs.m_accessor);
        return lhs.m_linear_index > rhs.m_linear_index;
    }

    friend bool operator>=(const accessor_iterator &lhs, const accessor_iterator &rhs) {
        SIMSYCL_CHECK(lhs.m_accessor == rhs.m_accessor);
        return lhs.m_linear_index >= rhs.m_linear_index;
    }

  private:
    template<typename, int, sycl::access_mode, sycl::target, sycl::access::placeholder>
    friend class sycl::accessor;
    template<typename, int, sycl::access_mode>
    friend class sycl::host_accessor;
    template<typename, int>
    friend class sycl::local_accessor;

    struct begin_t {
    } inline static constexpr begin{};
    struct end_t {
    } inline static constexpr end{};

    explicit accessor_iterator(const Accessor *accessor, begin_t /* tag */) : m_accessor(accessor), m_linear_index(0) {}

    explicit accessor_iterator(const Accessor *accessor, end_t /* tag */) : m_accessor(accessor) {
        if constexpr(Dimensions > 0) {
            m_linear_index = static_cast<difference_type>(accessor->get_range().size());
        } else {
            m_linear_index = 1;
        }
    }

    const Accessor *m_accessor = nullptr;
    difference_type m_linear_index;

    accessor_iterator &advance(const difference_type n) {
        SIMSYCL_CHECK(m_accessor != nullptr);
        SIMSYCL_CHECK(m_linear_index + n >= 0);
        if constexpr(Dimensions > 0) {
            SIMSYCL_CHECK(m_linear_index + n <= m_accessor->get_range().size());
        } else {
            SIMSYCL_CHECK(m_linear_index + n <= 1);
        }
        m_linear_index += n;
        return *this;
    }
};

} // namespace simsycl::detail

namespace simsycl::sycl::property {

struct no_init {};

} // namespace simsycl::sycl::property

namespace simsycl::sycl {

template<>
struct is_property<property::no_init> : std::true_type {};

template<typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget, access::placeholder IsPlaceholder>
struct is_property_of<property::no_init, accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder>>
    : std::true_type {};

template<typename DataT, int Dimensions>
struct is_property_of<property::no_init, local_accessor<DataT, Dimensions>> : std::true_type {};

template<typename DataT, int Dimensions, access_mode AccessMode>
struct is_property_of<property::no_init, host_accessor<DataT, Dimensions, AccessMode>> : std::true_type {};

inline constexpr simsycl::detail::accessor_tag<access_mode::read, target::device> read_only;
inline constexpr simsycl::detail::accessor_tag<access_mode::write, target::device> write_only;
inline constexpr simsycl::detail::accessor_tag<access_mode::read_write, target::device> read_write;
inline constexpr simsycl::detail::accessor_tag<access_mode::read, target::host_task> read_only_host_task;
inline constexpr simsycl::detail::accessor_tag<access_mode::write, target::host_task> write_only_host_task;
inline constexpr simsycl::detail::accessor_tag<access_mode::read_write, target::host_task> read_write_host_task;

inline constexpr property::no_init no_init;

template<typename DataT, int Dimensions, access_mode AccessMode, target AccessTarget, access::placeholder IsPlaceholder>
class accessor : public simsycl::detail::property_interface {
    static_assert(AccessMode == access_mode::read || !std::is_const_v<DataT>,
        "DataT must only be const-qualified when AccessMode == read");

  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    template<access::decorated IsDecorated>
    using accessor_ptr = std::enable_if_t<AccessTarget == target::device && IsDecorated == IsDecorated /* dependent */,
        multi_ptr<value_type, access::address_space::global_space, IsDecorated>>;

    using iterator = simsycl::detail::accessor_iterator<accessor, value_type, Dimensions>;
    using const_iterator = simsycl::detail::accessor_iterator<accessor, const value_type, Dimensions>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    accessor() = default;

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref,
        const detail::accessor_tag<AccessMode, AccessTarget> tag, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, tag, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        const detail::accessor_tag<AccessMode, AccessTarget> tag, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, tag, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        const property_list &prop_list = {})
        : accessor(internal, buffer_ref, access_range, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        const detail::accessor_tag<AccessMode, AccessTarget> tag, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, access_range, tag, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, access_range, access_offset, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, const detail::accessor_tag<AccessMode, AccessTarget> tag,
        const property_list &prop_list = {})
        : accessor(internal, buffer_ref, access_range, access_offset, tag, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, access_range, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, const detail::accessor_tag<AccessMode, AccessTarget> tag,
        const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, access_range, tag, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, id<Dimensions> access_offset, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, access_range, access_offset, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, id<Dimensions> access_offset,
        const detail::accessor_tag<AccessMode, AccessTarget> tag, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, access_range, access_offset, tag, prop_list) {}

    friend bool operator==(const accessor &lhs, const accessor &rhs) {
        return lhs.m_buffer == rhs.m_buffer && lhs.m_buffer_range == rhs.m_buffer_range
            && lhs.m_access_offset == rhs.m_access_offset && lhs.m_access_range == rhs.m_access_range
            && lhs.m_required == rhs.m_required;
    }

    friend bool operator!=(const accessor &lhs, const accessor &rhs) { return !(lhs == rhs); }

    void swap(accessor &other) { return std::swap(*this, other); }

    bool is_placeholder() const { return !m_required; }

    size_type byte_size() const noexcept {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_access_range.size() * sizeof(DataT);
    }

    size_type size() const noexcept {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_access_range.size();
    }

    size_type max_size() const noexcept { return std::numeric_limits<size_t>::max() / sizeof(DataT); }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_size() const { return byte_size(); }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_count() const { return size(); }

    bool empty() const noexcept { return m_access_range.size() == 0; }

    range<Dimensions> get_range() const {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_access_range;
    }

    id<Dimensions> get_offset() const {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_access_offset;
    }

    reference operator[](id<Dimensions> index) const
        requires(AccessMode != access_mode::atomic)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer[detail::get_linear_index(m_buffer_range, index)];
    }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL atomic<DataT, access::address_space::global_space> operator[](
        id<Dimensions> index) const
        requires(AccessMode == access_mode::atomic);

    decltype(auto) operator[](size_t index) const
        requires(Dimensions > 1)
    {
        return detail::subscript<Dimensions>(*this, index);
    }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL global_ptr<DataT> get_pointer() const noexcept
        requires(AccessTarget == target::device)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer;
    }

    std::add_pointer_t<value_type> get_pointer() const noexcept
        requires(AccessTarget == target::host_task)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer;
    }

    template<access::decorated IsDecorated>
    accessor_ptr<IsDecorated> get_multi_ptr() const noexcept
        requires(AccessTarget == target::device)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return accessor_ptr<IsDecorated>(m_buffer);
    }

    iterator begin() const noexcept { return iterator(this, iterator::begin); }

    iterator end() const noexcept { return iterator(this, iterator::end); }

    const_iterator cbegin() const noexcept { return const_iterator(this, const_iterator::begin); }

    const_iterator cend() const noexcept { return const_iterator(this, const_iterator::end); }

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;

  private:
    friend class handler;

    struct internal_t {
    } constexpr inline static internal{};

    DataT *m_buffer = nullptr;
    range<Dimensions> m_buffer_range;
    id<Dimensions> m_access_offset;
    range<Dimensions> m_access_range;
    // shared: require() on a copy is equivalent to require() on the original instance
    std::shared_ptr<bool> m_required = std::make_shared<bool>(false);

    template<typename AllocatorT>
    void init(buffer<DataT, Dimensions, AllocatorT> &buffer_ref) {
        m_buffer = detail::get_buffer_data(buffer_ref);
        m_buffer_range = buffer_ref.get_range();
        m_access_range = m_buffer_range;
    }
    void init(const id<Dimensions> &access_offset) { m_access_offset = access_offset; }

    void init(const range<Dimensions> &access_range) { m_access_range = access_range; }

    void init(handler & /* cgh */) { *m_required = true; }

    void init(const property_list &prop_list) {
        static_cast<detail::property_interface &>(*this)
            = detail::property_interface(prop_list, property_compatibility());
    }

    void init(simsycl::detail::accessor_tag<AccessMode, AccessTarget> /* tag */) {}

    template<typename... Params>
    explicit accessor(internal_t /* tag */, Params &&...args) {
        (init(args), ...);
    }

    void require() {
        SIMSYCL_CHECK(m_buffer != nullptr);
        *m_required = true;
    }

    const range<Dimensions> &get_buffer_range() const { return m_buffer_range; }
};

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, detail::accessor_tag<AccessMode, AccessTarget>,
    const property_list &) -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>,
    const property_list &) -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>,
    detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>,
    detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>, id<Dimensions>,
    detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>, id<Dimensions>,
    detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>, id<Dimensions>,
    detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>, id<Dimensions>,
    detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget, access::placeholder::false_t>;


template<typename DataT, access_mode AccessMode, target AccessTarget, access::placeholder IsPlaceholder>
class accessor<DataT, 0, AccessMode, AccessTarget, IsPlaceholder> : public simsycl::detail::property_interface {
    static_assert(AccessMode == access_mode::read || !std::is_const_v<DataT>,
        "DataT must only be const-qualified when AccessMode == read");

  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    template<access::decorated IsDecorated>
    using accessor_ptr = std::enable_if_t<AccessTarget == target::device,
        multi_ptr<value_type, access::address_space::global_space, IsDecorated>>;

    using iterator = simsycl::detail::accessor_iterator<accessor, value_type, 0>;
    using const_iterator = simsycl::detail::accessor_iterator<accessor, const value_type, 0>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    accessor() = default;

    template<typename AllocatorT>
    accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, const property_list &prop_list = {})
        : simsycl::detail::property_interface(prop_list, property_compatibility()),
          m_buffer(detail::get_buffer_data(buffer_ref)) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        const property_list &prop_list = {})
        : simsycl::detail::property_interface(prop_list, property_compatibility()),
          m_buffer(detail::get_buffer_data(buffer_ref)) {
        (void)command_group_handler_ref;
        *m_required = true;
    }

    friend bool operator==(const accessor &lhs, const accessor &rhs) {
        return lhs.m_buffer == rhs.m_buffer && lhs.m_required == rhs.m_required;
    }

    friend bool operator!=(const accessor &lhs, const accessor &rhs) { return !(lhs == rhs); }

    void swap(accessor &other) { return std::swap(*this, other); }

    bool is_placeholder() const { return !m_required; }

    size_type byte_size() const noexcept { return sizeof(DataT); }

    size_type size() const noexcept { return 1; }

    size_type max_size() const noexcept { return 1; }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_size() const { return byte_size(); }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_count() const { return size(); }

    bool empty() const noexcept { return false; }

    operator reference() const
        requires(AccessMode != access_mode::atomic)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return *m_buffer;
    }

    const accessor &operator=(const value_type &other) const
        requires(AccessMode != access_mode::atomic && AccessMode != access_mode::read)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        *m_buffer = other;
        return *this;
    }

    const accessor &operator=(value_type &&other) const
        requires(AccessMode != access_mode::atomic && AccessMode != access_mode::read)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        *m_buffer = std::move(other);
        return *this;
    }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL operator atomic<DataT, access::address_space::global_space>() const
        requires(AccessMode == access_mode::atomic);

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL global_ptr<DataT> get_pointer() const noexcept
        requires(AccessTarget == target::device)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer;
    }

    std::add_pointer_t<value_type> get_pointer() const noexcept
        requires(AccessTarget == target::host_task)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer;
    }

    template<access::decorated IsDecorated>
    accessor_ptr<IsDecorated> get_multi_ptr() const noexcept
        requires(AccessTarget == target::device)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return accessor_ptr<IsDecorated>(m_buffer);
    }


    iterator begin() const noexcept { return iterator(this, iterator::begin); }

    iterator end() const noexcept { return iterator(this, iterator::end); }

    const_iterator cbegin() const noexcept { return const_iterator(this, const_iterator::begin); }

    const_iterator cend() const noexcept { return const_iterator(this, const_iterator::end); }

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;

  private:
    friend class handler;

    DataT *m_buffer = nullptr;
    // shared: require() on a copy is equivalent to require() on the original instance
    std::shared_ptr<bool> m_required = std::make_shared<bool>(false);

    void require() {
        SIMSYCL_CHECK(m_buffer != nullptr);
        *m_required = true;
    }
};


template<typename DataT, int Dimensions>
class local_accessor final : public simsycl::detail::property_interface {
  private:
    using property_compatibility = detail::property_compatibility_with<local_accessor, property::no_init>;

  public:
    using value_type = DataT;
    using reference = value_type &;
    using const_reference = const DataT &;

    template<access::decorated IsDecorated>
    using accessor_ptr = multi_ptr<value_type, access::address_space::local_space, IsDecorated>;

    using iterator = simsycl::detail::accessor_iterator<local_accessor, value_type, Dimensions>;
    using const_iterator = simsycl::detail::accessor_iterator<local_accessor, const value_type, Dimensions>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    local_accessor() = default;

    local_accessor(
        range<Dimensions> allocation_size, handler &command_group_handler_ref, const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          m_allocation_ptr(detail::require_local_memory(
              command_group_handler_ref, allocation_size.size() * sizeof(DataT), alignof(DataT))),
          m_range(allocation_size) {}

    void swap(local_accessor &other) { std::swap(*this, other); }

    bool is_placeholder() const { return m_allocation_ptr == nullptr; }

    size_type byte_size() const noexcept { return m_range.size() * sizeof(DataT); }

    size_type size() const noexcept { return m_range.size(); }

    size_type max_size() const noexcept { return std::numeric_limits<size_t>::max() / sizeof(DataT); }

    bool empty() const noexcept { return m_range.size() == 0; }

    range<Dimensions> get_range() const { return m_range; }

    reference operator[](id<Dimensions> index) const {
        SIMSYCL_CHECK(get_allocation() != nullptr);
        return get_allocation()[detail::get_linear_index(m_range, index)];
    }

    decltype(auto) operator[](size_t index) const
        requires(Dimensions > 1)
    {
        return detail::subscript<Dimensions>(*this, index);
    }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL local_ptr<DataT> get_pointer() const noexcept {
        return local_ptr<DataT>(get_allocation());
    }

    template<access::decorated IsDecorated>
    accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
        return accessor_ptr<IsDecorated>(get_allocation());
    }

    iterator begin() const noexcept { return iterator(this, iterator::begin); }

    iterator end() const noexcept { return iterator(this, iterator::end); }

    const_iterator cbegin() const noexcept { return const_iterator(this, const_iterator::begin); }

    const_iterator cend() const noexcept { return const_iterator(this, const_iterator::end); }

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;

    friend bool operator==(const local_accessor &lhs, const local_accessor &rhs) {
        return lhs.m_allocation_ptr == rhs.m_allocation_ptr && lhs.m_range == rhs.m_range;
    }

    friend bool operator!=(const local_accessor &lhs, const local_accessor &rhs) { return !(lhs == rhs); }

  private:
    friend iterator;
    friend const_iterator;

    void **m_allocation_ptr = nullptr;
    sycl::range<Dimensions> m_range;

    const range<Dimensions> &get_buffer_range() const { return get_range(); }

    id<Dimensions> get_offset() const { return {}; }

    inline DataT *get_allocation() const {
        return m_allocation_ptr != nullptr ? static_cast<DataT *>(*m_allocation_ptr) : nullptr;
    }
};

template<typename DataT>
class local_accessor<DataT, 0> final : public simsycl::detail::property_interface {
  private:
    using property_compatibility = detail::property_compatibility_with<local_accessor, property::no_init>;

  public:
    using value_type = DataT;
    using reference = value_type &;
    using const_reference = const DataT &;

    template<access::decorated IsDecorated>
    using accessor_ptr = multi_ptr<value_type, access::address_space::local_space, IsDecorated>;

    using iterator = simsycl::detail::accessor_iterator<local_accessor, value_type, 0>;
    using const_iterator = simsycl::detail::accessor_iterator<local_accessor, const value_type, 0>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    local_accessor() = default;

    local_accessor(handler &command_group_handler_ref, const property_list &prop_list)
        : property_interface(prop_list, property_compatibility()),
          m_allocation_ptr(detail::require_local_memory(command_group_handler_ref, sizeof(DataT), alignof(DataT))) {}

    void swap(local_accessor &other) { std::swap(*this, other); }

    bool is_placeholder() const { return m_allocation_ptr == nullptr; }

    size_type byte_size() const noexcept { return sizeof(DataT); }

    size_type size() const noexcept { return 1; }

    size_type max_size() const noexcept { return 1; }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_size() const { return byte_size(); }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_count() const { return 1; }

    bool empty() const noexcept { return false; }

    operator reference() const {
        SIMSYCL_CHECK(get_allocation() != nullptr);
        return *get_allocation();
    }

    const local_accessor &operator=(const value_type &other) const {
        SIMSYCL_CHECK(get_allocation() != nullptr);
        *get_allocation() = other;
        return *this;
    }

    const local_accessor &operator=(value_type &&other) const {
        SIMSYCL_CHECK(get_allocation() != nullptr);
        *get_allocation() = std::move(other);
        return *this;
    }

    std::add_pointer_t<value_type> get_pointer() const noexcept { return get_allocation(); }

    template<access::decorated IsDecorated>
    accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
        return accessor_ptr<IsDecorated>(get_allocation());
    }

    iterator begin() const noexcept { return iterator(this, iterator::begin); }

    iterator end() const noexcept { return iterator(this, iterator::end); }

    const_iterator cbegin() const noexcept { return const_iterator(this, const_iterator::begin); }

    const_iterator cend() const noexcept { return const_iterator(this, const_iterator::end); }

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;

    friend bool operator==(const local_accessor &lhs, const local_accessor &rhs) {
        return lhs.m_allocation_ptr == rhs.m_allocation_ptr;
    }

    friend bool operator!=(const local_accessor &lhs, const local_accessor &rhs) { return !(lhs == rhs); }

  private:
    void **m_allocation_ptr = nullptr;

    inline DataT *get_allocation() const {
        return m_allocation_ptr != nullptr ? static_cast<DataT *>(*m_allocation_ptr) : nullptr;
    }
};


template<typename DataT, int Dimensions, access_mode AccessMode>
class host_accessor : public simsycl::detail::property_interface {
    static_assert(AccessMode == access_mode::read || !std::is_const_v<DataT>,
        "DataT must only be const-qualified when AccessMode == read");
    static_assert(
        AccessMode == access_mode::read || AccessMode == access_mode::read_write || AccessMode == access_mode::write,
        "host_accessor only supports read, read_write and write access modes");

  private:
    using property_compatibility = detail::property_compatibility_with<host_accessor, property::no_init>;

  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    using iterator = simsycl::detail::accessor_iterator<host_accessor, value_type, Dimensions>;
    using const_iterator = simsycl::detail::accessor_iterator<host_accessor, const value_type, Dimensions>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    host_accessor() = default;

    template<typename AllocatorT>
    host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, const property_list &prop_list = {})
        : host_accessor(internal, buffer_ref, prop_list) {}

    template<typename AllocatorT>
    host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref,
        detail::accessor_tag<AccessMode, target::device> tag, const property_list &prop_list = {})
        : host_accessor(internal, buffer_ref, tag, prop_list) {}

    template<typename AllocatorT>
    host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        const property_list &prop_list = {})
        : host_accessor(internal, buffer_ref, access_range, prop_list) {}

    template<typename AllocatorT>
    host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        detail::accessor_tag<AccessMode, target::device> tag, const property_list &prop_list = {})
        : host_accessor(internal, buffer_ref, access_range, tag, prop_list) {}

    template<typename AllocatorT>
    host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, const property_list &prop_list = {})
        : host_accessor(internal, buffer_ref, access_range, access_offset, prop_list) {}

    template<typename AllocatorT>
    host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, detail::accessor_tag<AccessMode, target::device> tag,
        const property_list &prop_list = {})
        : host_accessor(internal, buffer_ref, access_range, access_offset, tag, prop_list) {}

    friend bool operator==(const host_accessor &lhs, const host_accessor &rhs) {
        return lhs.m_buffer == rhs.m_buffer && lhs.m_buffer_range == rhs.m_buffer_range
            && lhs.m_access_offset == rhs.m_access_offset && lhs.m_access_range == rhs.m_access_range
            && lhs.m_required == rhs.m_required;
    }

    friend bool operator!=(const host_accessor &lhs, const host_accessor &rhs) { return !(lhs == rhs); }

    void swap(host_accessor &other) { std::swap(*this, other); }

    size_type byte_size() const noexcept { return m_access_range.size() * sizeof(DataT); }

    size_type size() const noexcept { return m_access_range.size(); }

    size_type max_size() const noexcept { return std::numeric_limits<size_t>::max() / sizeof(DataT); }

    bool empty() const noexcept { return m_access_range.size() == 0; }

    range<Dimensions> get_range() const { return m_access_range; }

    id<Dimensions> get_offset() const { return m_access_offset; }

    reference operator[](id<Dimensions> index) const
        requires(AccessMode != access_mode::atomic)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_buffer[detail::get_linear_index(m_buffer_range, index)];
    }

    decltype(auto) operator[](size_t index) const
        requires(Dimensions > 1)
    {
        return detail::subscript<Dimensions>(*this, index);
    }

    std::add_pointer_t<value_type> get_pointer() const noexcept { return m_buffer; }

    iterator begin() const noexcept { return iterator(this, iterator::begin); }

    iterator end() const noexcept { return iterator(this, iterator::end); }

    const_iterator cbegin() const noexcept { return const_iterator(this, const_iterator::begin); }

    const_iterator cend() const noexcept { return const_iterator(this, const_iterator::end); }

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;

  private:
    friend class handler;

    struct internal_t {
    } constexpr inline static internal{};

    DataT *m_buffer = nullptr;
    range<Dimensions> m_buffer_range;
    id<Dimensions> m_access_offset;
    range<Dimensions> m_access_range;

    template<typename AllocatorT>
    void init(buffer<DataT, Dimensions, AllocatorT> &buffer_ref) {
        m_buffer = detail::get_buffer_data(buffer_ref);
        m_buffer_range = buffer_ref.get_range();
        m_access_range = m_buffer_range;
    }
    void init(const id<Dimensions> &access_offset) { m_access_offset = access_offset; }

    void init(const range<Dimensions> &access_range) { m_access_range = access_range; }

    void init(const property_list &prop_list) {
        static_cast<detail::property_interface &>(*this)
            = detail::property_interface(prop_list, detail::property_compatibility());
    }

    void init(simsycl::detail::accessor_tag<AccessMode, target::device> /* tag */) {}

    template<typename... Params>
    explicit host_accessor(internal_t /* tag */, Params &&...args) {
        (init(args), ...);
    }

    const range<Dimensions> &get_buffer_range() const { return m_buffer_range; }
};

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
    detail::accessor_tag<AccessMode, target::device> tag, const property_list &prop_list = {})
    -> host_accessor<DataT, Dimensions, AccessMode>;

template<typename DataT, int Dimensions, typename AllocatorT, access_mode AccessMode>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
    id<Dimensions> access_offset, detail::accessor_tag<AccessMode, target::device> tag,
    const property_list &prop_list = {}) -> host_accessor<DataT, Dimensions, AccessMode>;

template<typename DataT, access_mode AccessMode>
class host_accessor<DataT, 0, AccessMode> : public simsycl::detail::property_interface {
    static_assert(AccessMode == access_mode::read || !std::is_const_v<DataT>,
        "DataT must only be const-qualified when AccessMode == read");
    static_assert(
        AccessMode == access_mode::read || AccessMode == access_mode::read_write || AccessMode == access_mode::write,
        "host_accessor only supports read, read_write and write access modes");

  private:
    using property_compatibility = detail::property_compatibility_with<host_accessor, property::no_init>;

  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    using iterator = simsycl::detail::accessor_iterator<host_accessor, value_type, 0>;
    using const_iterator = simsycl::detail::accessor_iterator<host_accessor, const value_type, 0>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = size_t;

    host_accessor();

    template<typename AllocatorT>
    host_accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, const property_list &prop_list = {});

    friend bool operator==(const host_accessor &lhs, const host_accessor &rhs) = default;

    void swap(host_accessor &other) { return std::swap(*this, other); }

    size_type byte_size() const noexcept { return sizeof(DataT); }

    size_type size() const noexcept { return 1; }

    size_type max_size() const noexcept { return 1; }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_size() const { return byte_size(); }

    SIMSYCL_DETAIL_DEPRECATED_IN_SYCL size_t get_count() const { return size(); }

    bool empty() const noexcept { return false; }

    operator reference() const
        requires(AccessMode != access_mode::atomic)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return *m_buffer;
    }

    const host_accessor &operator=(const value_type &other) const
        requires(AccessMode != access_mode::atomic && AccessMode != access_mode::read)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        *m_buffer = other;
        return *this;
    }

    const host_accessor &operator=(value_type &&other) const
        requires(AccessMode != access_mode::atomic && AccessMode != access_mode::read)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        *m_buffer = std::move(other);
        return *this;
    }

    std::add_pointer_t<value_type> get_pointer() const noexcept { return m_buffer; }

    iterator begin() const noexcept { return iterator(this, iterator::begin); }

    iterator end() const noexcept { return iterator(this, iterator::end); }

    const_iterator cbegin() const noexcept { return const_iterator(this, const_iterator::begin); }

    const_iterator cend() const noexcept { return const_iterator(this, const_iterator::end); }

    reverse_iterator rbegin() const noexcept;

    reverse_iterator rend() const noexcept;

    const_reverse_iterator crbegin() const noexcept;

    const_reverse_iterator crend() const noexcept;

  private:
    friend class handler;

    DataT *m_buffer = nullptr;
};


// deprecated accessor specializations

template<typename DataT, int Dimensions, access_mode AccessMode, access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, target::constant_buffer, IsPlaceholder> final
    : public simsycl::detail::property_interface {
    static_assert(
        AccessMode == access_mode::read, "accessor<constant_buffer> is only available for AccessMode == read");

  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = const DataT;
    using reference = const DataT &;
    using const_reference = const DataT &;

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        const property_list &prop_list = {})
        : accessor(internal, buffer_ref, access_range, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, access_range, access_offset, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, access_range, prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        range<Dimensions> access_range, id<Dimensions> access_offset, const property_list &prop_list = {})
        : accessor(internal, buffer_ref, command_group_handler_ref, access_range, access_offset, prop_list) {}

    friend bool operator==(const accessor &lhs, const accessor &rhs) = default;

    bool is_placeholder() const { return !*m_required; }

    size_t get_size() const noexcept { return get_count() * sizeof(DataT); }

    size_t get_count() const noexcept { return m_access_range.size(); }

    range<Dimensions> get_range() const {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_access_range;
    }

    id<Dimensions> get_offset() const {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_access_offset;
    }

    reference operator[](id<Dimensions> index) const
        requires(AccessMode != access_mode::atomic)
    {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer[detail::get_linear_index(m_buffer_range, index)];
    }

    decltype(auto) operator[](size_t index) const
        requires(Dimensions > 1)
    {
        return detail::subscript<Dimensions>(*this, index);
    }

    constant_ptr<DataT> get_pointer() const noexcept {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer;
    }

  private:
    struct internal_t {
    } constexpr inline static internal{};

    DataT *m_buffer = nullptr;
    range<Dimensions> m_buffer_range;
    id<Dimensions> m_access_offset;
    range<Dimensions> m_access_range;
    // shared: require() on a copy is equivalent to require() on the original instance
    std::shared_ptr<bool> m_required = std::make_shared<bool>(false);

    template<typename AllocatorT>
    void init(buffer<DataT, Dimensions, AllocatorT> &buffer_ref) {
        m_buffer = detail::get_buffer_data(buffer_ref);
        m_buffer_range = buffer_ref.get_range();
        m_access_range = m_buffer_range;
    }
    void init(const id<Dimensions> &access_offset) { m_access_offset = access_offset; }

    void init(const range<Dimensions> &access_range) { m_access_range = access_range; }

    void init(handler & /* cgh */) { *m_required = true; }

    void init(const property_list &prop_list) {
        static_cast<detail::property_interface &>(*this)
            = detail::property_interface(prop_list, property_compatibility());
    }

    template<typename... Params>
    explicit accessor(internal_t /* tag */, Params &&...args) {
        (init(args), ...);
    }

    void require() {
        SIMSYCL_CHECK(m_buffer != nullptr);
        *m_required = true;
    }

    const range<Dimensions> &get_buffer_range() const { return m_buffer_range; }
};

template<typename DataT, access_mode AccessMode, access::placeholder IsPlaceholder>
class accessor<DataT, 0, AccessMode, target::constant_buffer, IsPlaceholder> final
    : public simsycl::detail::property_interface {
    static_assert(
        AccessMode == access_mode::read, "accessor<constant_buffer> is only available for AccessMode == read");

  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = const DataT;
    using reference = const DataT &;
    using const_reference = const DataT &;

    template<typename AllocatorT>
    accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, const property_list &prop_list = {})
        : simsycl::detail::property_interface(prop_list, property_compatibility()),
          m_buffer(detail::get_buffer_data(buffer_ref)) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, handler &command_group_handler_ref,
        const property_list &prop_list = {})
        : simsycl::detail::property_interface(prop_list, property_compatibility()),
          m_buffer(detail::get_buffer_data(buffer_ref)) {
        (void)command_group_handler_ref;
        *m_required = true;
    }

    friend bool operator==(const accessor &lhs, const accessor &rhs) = default;

    bool is_placeholder() const { return !m_required; }

    size_t get_size() const noexcept { return sizeof(DataT); }

    size_t get_count() const noexcept { return 1; }

    operator reference() const {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return *m_buffer;
    }

    global_ptr<DataT> get_pointer() const noexcept {
        SIMSYCL_CHECK(m_buffer != nullptr);
        SIMSYCL_CHECK(*m_required);
        return m_buffer;
    }

  private:
    friend class handler;

    DataT *m_buffer = nullptr;
    // shared: require() on a copy is equivalent to require() on the original instance
    std::shared_ptr<bool> m_required = std::make_shared<bool>(false);

    void require() {
        SIMSYCL_CHECK(m_buffer != nullptr);
        *m_required = true;
    }
};

template<typename DataT, int Dimensions, access_mode AccessMode, access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, target::host_buffer, IsPlaceholder> final
    : public simsycl::detail::property_interface {
    static_assert(AccessMode == access_mode::read || !std::is_const_v<DataT>,
        "DataT must only be const-qualified when AccessMode == read");
    static_assert(AccessMode != access_mode::atomic, "accessor<..., host_buffer> is not available for atomic access");

  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, const property_list &prop_list = {})
        : accessor(buffer_ref, buffer_ref.get_range(), id<Dimensions>(), prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        const property_list &prop_list = {})
        : accessor(buffer_ref, access_range, id<Dimensions>(), prop_list) {}

    template<typename AllocatorT>
    accessor(buffer<DataT, Dimensions, AllocatorT> &buffer_ref, range<Dimensions> access_range,
        id<Dimensions> access_offset, const property_list &prop_list = {})
        : detail::property_interface(prop_list, property_compatibility()),
          m_buffer(detail::get_buffer_data(buffer_ref)), m_buffer_range(buffer_ref.get_range()),
          m_access_range(access_range), m_access_offset(access_offset) {}

    bool is_placeholder() const { return false; }

    size_t get_size() const { return get_count() * sizeof(DataT); }

    size_t get_count() const { return m_access_range.size(); }

    range<Dimensions> get_range() const { return m_access_range; }

    id<Dimensions> get_offset() const { return m_access_offset; }

    reference operator[](id<Dimensions> index) const {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return m_buffer[detail::get_linear_index(m_buffer_range, index)];
    }

    decltype(auto) operator[](size_t index) const
        requires(Dimensions > 1)
    {
        return detail::subscript<Dimensions>(*this, index);
    }

    std::add_pointer_t<value_type> get_pointer() const noexcept { return m_buffer; }

    friend bool operator==(const accessor &lhs, const accessor &rhs) {
        return lhs.m_buffer == rhs.m_buffer && lhs.m_buffer_range == rhs.m_buffer_range
            && lhs.m_access_offset == rhs.m_access_offset && lhs.m_access_range == rhs.m_access_range
            && lhs.m_required == rhs.m_required;
    }

    friend bool operator!=(const accessor &lhs, const accessor &rhs) { return !(lhs == rhs); }

  private:
    DataT *m_buffer = nullptr;
    range<Dimensions> m_buffer_range;
    id<Dimensions> m_access_offset;
    range<Dimensions> m_access_range;
};

template<typename DataT, access_mode AccessMode, access::placeholder IsPlaceholder>
class accessor<DataT, 0, AccessMode, target::host_buffer, IsPlaceholder> {
    static_assert(AccessMode == access_mode::read || !std::is_const_v<DataT>,
        "DataT must only be const-qualified when AccessMode == read");
    static_assert(AccessMode != access_mode::atomic, "accessor<..., host_buffer> is not available for atomic access");

  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
    using reference = value_type &;
    using const_reference = const DataT &;

    template<typename AllocatorT>
    accessor(buffer<DataT, 1, AllocatorT> &buffer_ref, const property_list &prop_list = {})
        : detail::property_interface(prop_list, property_compatibility()),
          m_buffer(detail::get_buffer_data(buffer_ref)) {}

    bool is_placeholder() const { return false; }

    size_t get_size() const { return sizeof(DataT); }

    size_t get_count() const { return 1; }

    operator reference() const {
        SIMSYCL_CHECK(m_buffer != nullptr);
        return *m_buffer;
    }

    std::add_pointer_t<value_type> get_pointer() const noexcept { return m_buffer; }

    friend bool operator==(const accessor &lhs, const accessor &rhs) {
        return lhs.m_buffer == rhs.m_buffer && lhs.m_required == rhs.m_required;
    }

    friend bool operator!=(const accessor &lhs, const accessor &rhs) { return !(lhs == rhs); }

  private:
    DataT *m_buffer = nullptr;
};

template<typename DataT, int Dimensions, access_mode AccessMode, access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, target::local, IsPlaceholder> final
    : public simsycl::detail::property_interface {
  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = DataT;
    using reference = DataT &;
    using const_reference = const DataT &;

    accessor(range<Dimensions> allocation_size, handler &command_group_handler_ref, const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          m_allocation_ptr(detail::require_local_memory(
              command_group_handler_ref, allocation_size.size() * sizeof(DataT), alignof(DataT))),
          m_range(allocation_size) {}

    size_t get_size() const { return m_range.size() * sizeof(DataT); }

    size_t get_count() const { return m_range.size(); }

    range<Dimensions> get_range() const { return m_range; }

    reference operator[](id<Dimensions> index) const
        requires(AccessMode == access_mode::read_write)
    {
        SIMSYCL_CHECK(get_allocation() != nullptr);
        return get_allocation()[detail::get_linear_index(m_range, index)];
    }

    decltype(auto) operator[](size_t index) const
        requires(Dimensions > 1)
    {
        return detail::subscript<Dimensions>(*this, index);
    }

    atomic<DataT, access::address_space::local_space> operator[](id<Dimensions> index) const
        requires(AccessMode == access_mode::atomic);

    atomic<DataT, access::address_space::local_space> operator[](size_t index) const
        requires(AccessMode == access_mode::atomic);

    local_ptr<DataT> get_pointer() const noexcept { return local_ptr<DataT>(get_allocation()); }

    friend bool operator==(const accessor &lhs, const accessor &rhs) {
        return lhs.m_allocation_ptr == rhs.m_allocation_ptr && lhs.m_range == rhs.m_range;
    }

    friend bool operator!=(const accessor &lhs, const accessor &rhs) { return !(lhs == rhs); }

  private:
    void **m_allocation_ptr;
    sycl::range<Dimensions> m_range;

    const range<Dimensions> &get_buffer_range() const { return get_range(); }

    inline DataT *get_allocation() const { return static_cast<DataT *>(*m_allocation_ptr); }
};

template<typename DataT, access_mode AccessMode, access::placeholder IsPlaceholder>
class accessor<DataT, 0, AccessMode, target::local, IsPlaceholder> final : public simsycl::detail::property_interface {
  private:
    using property_compatibility = detail::property_compatibility_with<accessor, property::no_init>;

  public:
    using value_type = DataT;
    using reference = DataT &;
    using const_reference = const DataT &;

    accessor(handler &command_group_handler_ref, const property_list &prop_list = {})
        : property_interface(prop_list, property_compatibility()),
          m_allocation_ptr(detail::require_local_memory(command_group_handler_ref, sizeof(DataT), alignof(DataT))) {}

    size_t get_size() const { return sizeof(DataT); }

    size_t get_count() const { return 1; }

    operator reference() const
        requires(AccessMode == access_mode::read_write)
    {
        SIMSYCL_CHECK(get_allocation() != nullptr);
        return *get_allocation();
    }

    operator atomic<DataT, access::address_space::local_space>() const
        requires(AccessMode == access_mode::atomic);

    local_ptr<DataT> get_pointer() const noexcept { return local_ptr<DataT>(get_allocation()); }

  private:
    void **m_allocation_ptr;

    inline DataT *get_allocation() const { return static_cast<DataT *>(*m_allocation_ptr); }
};


SIMSYCL_STOP_IGNORING_DEPRECATIONS


} // namespace simsycl::sycl
