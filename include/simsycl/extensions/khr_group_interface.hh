#include "sycl/sycl.hpp" // IWYU pragma: keep

#define SYCL_KHR_GROUP_INTERFACE 1

namespace simsycl::sycl::khr {

template<typename ParentGroup>
class member_item {
  public:
    using id_type = typename ParentGroup::id_type;
    using linear_id_type = typename ParentGroup::linear_id_type;
    using range_type = typename ParentGroup::range_type;
    // using extents_type = /* extents of all 1s with ParentGroup's index type */; // C++23
    using size_type = typename ParentGroup::size_type;
    static constexpr int dimensions = ParentGroup::dimensions;
    static constexpr memory_scope fence_scope = memory_scope::work_item;

    /* -- common by-value interface members -- */

    id_type id() const noexcept { return m_parent_group.get_local_id(); }
    linear_id_type linear_id() const noexcept { return m_parent_group.get_local_linear_id(); }

    range_type range() const noexcept { return m_parent_group.get_local_range(); }

    // constexpr extents_type extents() const noexcept;                                     // C++23
    // constexpr extents_type::index_type extent(extents_type::rank_type r) const noexcept; // C++23

    // static constexpr extents_type::rank_type rank() noexcept;         // C++23
    // static constexpr extents_type::rank_type rank_dynamic() noexcept; // C++23
    // static constexpr size_t static_extent(rank_type r) noexcept;      // C++23

    constexpr size_type size() const noexcept { return 1; }

  private:
    ParentGroup m_parent_group;
    member_item(ParentGroup g) noexcept : m_parent_group(g) {}

    linear_id_type get_local_linear_id() const noexcept { return m_parent_group.get_local_linear_id(); }

    template<typename Group>
    friend member_item<Group> get_member_item(Group g) noexcept;
    template<typename Group>
    friend bool leader_of(Group g) noexcept;
};

template<int Dimensions = 1>
class work_group {
  public:
    using id_type = sycl::id<Dimensions>;
    using linear_id_type = size_t;
    using range_type = sycl::range<Dimensions>;
    // using extents_type = std::dextents<size_t, Dimensions>; // C++23
    using size_type = size_t;
    static constexpr int dimensions = Dimensions;
    static constexpr memory_scope fence_scope = memory_scope::work_group;

    work_group(group<Dimensions> g) noexcept : m_group(g) {}

    operator group<Dimensions>() const noexcept { return m_group; }

    /* -- common by-value interface members -- */

    id_type id() const noexcept { return m_group.get_group_id(); }
    linear_id_type linear_id() const noexcept { return m_group.get_group_linear_id(); }

    range_type range() const noexcept { return m_group.get_group_range(); }

    // extents_type extents() const noexcept;                                     // C++23
    // extents_type::index_type extent(extents_type::rank_type r) const noexcept; // C++23

    // static constexpr extents_type::rank_type rank() noexcept;         // C++23
    // static constexpr extents_type::rank_type rank_dynamic() noexcept; // C++23
    // static constexpr size_t static_extent(rank_type r) noexcept;      // C++23

    size_type size() const noexcept { return m_group.get_local_range().size(); }

  private:
    group<Dimensions> m_group;

    id_type get_local_id() const noexcept { return m_group.get_local_id(); }
    linear_id_type get_local_linear_id() const noexcept { return m_group.get_local_linear_id(); }
    range_type get_local_range() const noexcept { return m_group.get_local_range(); }
    friend class member_item<work_group>;
    template<typename Group>
    friend bool leader_of(Group g) noexcept;
};

class sub_group {
  public:
    using id_type = sycl::id<1>;
    using linear_id_type = uint32_t;
    using range_type = sycl::range<1>;
    // using extents_type = std::dextents<uint32_t, 1>; // C++23
    using size_type = uint32_t;
    static constexpr int dimensions = 1;
    static constexpr memory_scope fence_scope = memory_scope::sub_group;

    sub_group(sycl::sub_group sg) noexcept : m_sub_group(sg) {}

    operator sycl::sub_group() const noexcept { return m_sub_group; }

    /* -- common by-value interface members -- */

    id_type id() const noexcept { return m_sub_group.get_group_id(); }
    linear_id_type linear_id() const noexcept { return m_sub_group.get_group_linear_id(); }

    range_type range() const noexcept { return m_sub_group.get_group_range(); }

    // extents_type extents() const noexcept;                                     // C++23
    // extents_type::index_type extent(extents_type::rank_type r) const noexcept; // C++23

    // static constexpr extents_type::rank_type rank() noexcept;         // C++23
    // static constexpr extents_type::rank_type rank_dynamic() noexcept; // C++23
    // static constexpr size_t static_extent(rank_type r) noexcept;      // C++23

    size_type size() const noexcept { return m_sub_group.get_local_range().size(); }
    size_type max_size() const noexcept { return m_sub_group.get_max_local_range().size(); }

  private:
    sycl::sub_group m_sub_group;

    id_type get_local_id() const noexcept { return m_sub_group.get_local_id(); }
    linear_id_type get_local_linear_id() const noexcept { return m_sub_group.get_local_linear_id(); }
    range_type get_local_range() const noexcept { return m_sub_group.get_local_range(); }
    friend class member_item<sub_group>;
    template<typename Group>
    friend bool leader_of(Group g) noexcept;
};

template<typename Group>
member_item<Group> get_member_item(Group g) noexcept {
    return member_item<Group>(g);
}

template<typename Group>
bool leader_of(Group g) noexcept {
    return g.get_local_linear_id() == 0;
}

} // namespace simsycl::sycl::khr
