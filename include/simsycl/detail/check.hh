#pragma once

#include <simsycl/config.hh>

#include <source_location>

#define SIMSYCL_CHECK_NONE 1
#define SIMSYCL_CHECK_LOG 2
#define SIMSYCL_CHECK_THROW 3
#define SIMSYCL_CHECK_ABORT 4

namespace simsycl::detail {
void check_log(bool condition, const char *cond_string, std::source_location location);
void check_throw(bool condition, const char *cond_string, std::source_location location);
void check_abort(bool condition, const char *cond_string, std::source_location location);

struct sink {
    template<typename... Args>
    sink(const Args &...) {} // NOLINT
};
} // namespace simsycl::detail

#if SIMSYCL_CHECK_MODE == SIMSYCL_CHECK_NONE
#define SIMSYCL_CHECK(CONDITION)                                                                                       \
    do { (void)(CONDITION); } while(0)
#elif SIMSYCL_CHECK_MODE == SIMSYCL_CHECK_LOG
#define SIMSYCL_CHECK(CONDITION)                                                                                       \
    do { simsycl::detail::check_log(CONDITION, #CONDITION, std::source_location::current()); } while(0)
#elif SIMSYCL_CHECK_MODE == SIMSYCL_CHECK_THROW
#define SIMSYCL_CHECK(CONDITION)                                                                                       \
    do { simsycl::detail::check_throw(CONDITION, #CONDITION, std::source_location::current()); } while(0)
#elif SIMSYCL_CHECK_MODE == SIMSYCL_CHECK_ABORT
#define SIMSYCL_CHECK(CONDITION)                                                                                       \
    do { simsycl::detail::check_abort(CONDITION, #CONDITION, std::source_location::current()); } while(0)
#else
#error "SIMSYCL_CHECK_MODE must be SIMSYCL_CHECK_NONE, SIMSYCL_CHECK_LOG, SIMSYCL_CHECK_THROW, or SIMSYCL_CHECK_ABORT"
#endif

#define SIMSYCL_NOT_IMPLEMENTED                                                                                        \
    printf("SIMSYCL: Not implemented (%s:%d)\n", __FILE__, __LINE__);                                                  \
    abort();

#define SIMSYCL_NOT_IMPLEMENTED_UNUSED_ARGS(...)                                                                       \
    simsycl::detail::sink{__VA_ARGS__};                                                                                \
    SIMSYCL_NOT_IMPLEMENTED
