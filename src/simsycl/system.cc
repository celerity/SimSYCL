#include "simsycl/system.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/platform.hh"
#include "simsycl/templates.hh"


namespace simsycl {

system_config system = [] {
    system_config system; // gpuc3
    auto platform = system.platforms.emplace_back(create_platform(simsycl::templates::platform::cuda_12_2));
    for(int i = 0; i < 4; ++i) {
        system.devices.push_back(create_device(platform, simsycl::templates::device::nvidia::rtx_3090));
    }
    return system;
}(); // IIFE

}

namespace simsycl::detail {

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

} // namespace simsycl::detail

namespace simsycl::sycl {

std::error_code make_error_code(errc e) noexcept { return {static_cast<int>(e), detail::error_category_v}; }

const std::error_category &sycl_category() noexcept { return detail::error_category_v; }

} // namespace simsycl::sycl
