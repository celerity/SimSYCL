#include "simsycl/detail/check.hh"
#include "simsycl/sycl/exception.hh"

#include <iostream>

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

void check_log(bool condition, const char *cond_string, std::source_location location) {
    if(!condition) { std::cout << format_error(cond_string, location).c_str(); }
}

void check_throw(bool condition, const char *cond_string, std::source_location location) {
    if(!condition) { throw simsycl::sycl::exception(sycl::errc::invalid, format_error(cond_string, location)); }
}

void check_abort(bool condition, const char *cond_string, std::source_location location) {
    if(!condition) {
        std::cout << format_error(cond_string, location).c_str();
        abort();
    }
}

} // namespace simsycl::detail