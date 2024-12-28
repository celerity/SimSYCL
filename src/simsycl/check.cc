#include "simsycl/detail/check.hh"
#include "simsycl/sycl/exception.hh"
#include "simsycl/sycl/property.hh"

// TODO: use std::format/print once widely available
#include <cassert>
#include <iostream>
#include <stdarg.h>

namespace {
std::string format_error(const char *cond_string, std::source_location location) {
    // TODO: use std::format when it's available on all target platforms
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "SimSYCL check failed: %s at %s:%u:%u\n", cond_string, location.file_name(),
        location.line(), location.column());
    return buffer;
}
} // namespace

namespace simsycl::detail {

constexpr int no_check_override = 0;
thread_local int g_check_mode_override = no_check_override;

override_check_mode::override_check_mode(int mode) {
    assert(g_check_mode_override == no_check_override && "check mode already overridden");
    g_check_mode_override = mode;
}
override_check_mode::~override_check_mode() { g_check_mode_override = no_check_override; }

void check(bool condition, const char *cond_string, std::source_location location, int default_mode,
    const char *message, ...) {
    int mode = default_mode;
    if(g_check_mode_override != no_check_override) { mode = g_check_mode_override; }
    if(!condition) {
        char buffer[4096];
        va_list args;
        va_start(args, message);
        vsnprintf(buffer, sizeof(buffer), message, args);
        va_end(args);
        switch(mode) {
            case SIMSYCL_CHECK_LOG:
                std::cout << format_error(cond_string, location).c_str() << buffer << std::endl;
                break;
            case SIMSYCL_CHECK_THROW:
                throw simsycl::sycl::exception(sycl::errc::invalid, format_error(cond_string, location) + buffer);
            case SIMSYCL_CHECK_ABORT:
                std::cout << format_error(cond_string, location).c_str() << buffer << std::endl;
                abort();
            default: assert(false && "invalid check mode");
        }
    }
}

void throw_invalid_property() {
    throw simsycl::sycl::exception(sycl::errc::invalid, "object does not hold requested property");
}

} // namespace simsycl::detail