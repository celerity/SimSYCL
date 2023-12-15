#include "simsycl/system.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/platform.hh"
#include "simsycl/templates.hh"

#include <limits>

namespace simsycl::detail {

sycl::device select_device(const device_selector &selector) {
    auto &system = simsycl::get_system();
    SIMSYCL_CHECK(!system.devices.empty());
    int max_rating = std::numeric_limits<int>::lowest();
    for(const auto &device : system.devices) {
        if(int rating = selector(device); rating > max_rating) { max_rating = rating; }
    }
    if(max_rating < 0) { throw sycl::exception(sycl::errc::runtime, "No suitable device found"); }
    const auto device = std::find_if(system.devices.begin(), system.devices.end(),
        [&](const auto &device) { return selector(device) == max_rating; });
    assert(device != system.devices.end());
    return *device;
}

class error_category : public std::error_category {
    const char *name() const noexcept override { return "sycl"; }

    std::string message(int condition) const override {
        switch(static_cast<sycl::errc>(condition)) {
            case sycl::errc::success: return "success";
            case sycl::errc::runtime: return "runtime";
            case sycl::errc::kernel: return "kernel";
            case sycl::errc::accessor: return "accessor";
            case sycl::errc::nd_range: return "nd_range";
            case sycl::errc::event: return "event";
            case sycl::errc::kernel_argument: return "kernel argument";
            case sycl::errc::build: return "build";
            case sycl::errc::invalid: return "invalid";
            case sycl::errc::memory_allocation: return "memory allocation";
            case sycl::errc::platform: return "platform";
            case sycl::errc::profiling: return "profiling";
            case sycl::errc::feature_not_supported: return "feature not supported";
            case sycl::errc::kernel_not_supported: return "kernel not supported";
            case sycl::errc::backend_mismatch: return "backend mismatch";
            default: return "unknown";
        }
    }
};

const error_category error_category_v;

void default_async_handler(sycl::exception_list exceptions) {
    fprintf(stderr, "SimSYCL default async exception handler called for\n");
    for(const auto &exception : exceptions) {
        try {
            std::rethrow_exception(exception);
        } catch(const sycl::exception &e) { //
            fprintf(stderr, "  - sycl::exception: %s\n", e.what());
        } catch(const std::exception &e) { //
            fprintf(stderr, "  - std::exception: %s\n", e.what());
        } catch(...) { //
            fprintf(stderr, "  - unknown exception\n");
        }
    }
    fprintf(stderr, "terminating.\n");
    std::terminate();
}

void call_async_handler(const sycl::async_handler &handler_opt, sycl::exception_list exceptions) {
    handler_opt ? handler_opt(exceptions) : default_async_handler(exceptions);
}

std::optional<system_config> system;

} // namespace simsycl::detail

namespace simsycl::sycl {

std::error_code make_error_code(errc e) noexcept { return {static_cast<int>(e), detail::error_category_v}; }

const std::error_category &sycl_category() noexcept { return detail::error_category_v; }

} // namespace simsycl::sycl

namespace simsycl {

const system_config &get_system() {
    if(!detail::system.has_value()) {
        auto &system = detail::system.emplace(); // gpuc3
        auto platform = system.platforms.emplace_back(create_platform(simsycl::templates::platform::cuda_12_2));
        for(int i = 0; i < 4; ++i) {
            system.devices.push_back(create_device(platform, simsycl::templates::device::nvidia::rtx_3090));
        }
    }
    return *detail::system;
}

void set_system(system_config system) { detail::system = std::move(system); }

} // namespace simsycl
