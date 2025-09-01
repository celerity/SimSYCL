#include <simsycl/sycl/group.hh>
#include <simsycl/sycl/nd_item.hh>
#include <simsycl/sycl/sub_group.hh>

namespace simsycl::sycl::khr {

#if SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES

namespace detail {
template<int Dimensions>
thread_local std::optional<simsycl::sycl::nd_item<Dimensions>> g_khr_wi_query_this_nd_item;

template<int Dimensions>
thread_local std::optional<simsycl::sycl::group<Dimensions>> g_khr_wi_query_this_group;

inline thread_local std::optional<simsycl::sycl::sub_group> g_khr_wi_query_this_sub_group;

inline void khr_wi_query_check(bool val, [[maybe_unused]] const char *query_name) {
    SIMSYCL_CHECK_MSG(val,
        "Work item query state '%s' is not available.\n"
        "Make sure that the query originated from a kernel launched with a sycl::nd_range argument",
        query_name);
}

} // namespace detail

template<int Dimensions>
simsycl::sycl::nd_item<Dimensions> this_nd_item() {
    detail::khr_wi_query_check(detail::g_khr_wi_query_this_nd_item<Dimensions>.has_value(), "this_nd_item");
    return detail::g_khr_wi_query_this_nd_item<Dimensions>.value();
}

template<int Dimensions>
simsycl::sycl::group<Dimensions> this_group() {
    detail::khr_wi_query_check(detail::g_khr_wi_query_this_group<Dimensions>.has_value(), "this_group");
    return detail::g_khr_wi_query_this_group<Dimensions>.value();
}

inline simsycl::sycl::sub_group this_sub_group() {
    detail::khr_wi_query_check(detail::g_khr_wi_query_this_sub_group.has_value(), "this_sub_group");
    return detail::g_khr_wi_query_this_sub_group.value();
}

#endif // SIMSYCL_ENABLE_SYCL_KHR_WORK_ITEM_QUERIES

} // namespace simsycl::sycl::khr
