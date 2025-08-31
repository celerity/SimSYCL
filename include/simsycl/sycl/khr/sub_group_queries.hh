#include <simsycl/sycl/group.hh>
#include <simsycl/sycl/nd_item.hh>
#include <simsycl/sycl/sub_group.hh>

namespace simsycl::sycl::khr {

#if SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES

namespace detail {
template<int Dimensions>
std::optional<simsycl::sycl::nd_item<Dimensions>> g_khr_wi_query_this_nd_item;

template<int Dimensions>
std::optional<simsycl::sycl::group<Dimensions>> g_khr_wi_query_this_group;

inline std::optional<simsycl::sycl::sub_group> g_khr_wi_query_this_sub_group;
} // namespace detail

template<int Dimensions>
simsycl::sycl::nd_item<Dimensions> this_nd_item() {
    SIMSYCL_CHECK_MSG(!!detail::g_khr_wi_query_this_nd_item<Dimensions>,
        "Work item query state 'this_nd_item' is not available.\n"
        "Make sure that the query originated from a kernel launched with a sycl::nd_range argument");
    return detail::g_khr_wi_query_this_nd_item<Dimensions>.value();
}

template<int Dimensions>
simsycl::sycl::group<Dimensions> this_group() {
    SIMSYCL_CHECK_MSG(!!detail::g_khr_wi_query_this_group<Dimensions>,
        "Work item query state 'this_group' is not available.\n"
        "Make sure that the query originated from a kernel launched with a sycl::nd_range argument");
    return detail::g_khr_wi_query_this_group<Dimensions>.value();
}

inline simsycl::sycl::sub_group this_sub_group() {
    SIMSYCL_CHECK_MSG(!!detail::g_khr_wi_query_this_sub_group,
        "Work item query state 'this_sub_group' is not available.\n"
        "Make sure that the query originated from a kernel launched with a sycl::nd_range argument");
    return detail::g_khr_wi_query_this_sub_group.value();
}

#endif // SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES

} // namespace simsycl::sycl::khr
