#include "simsycl/system.hh"
#include "simsycl/detail/allocation.hh"
#include "simsycl/detail/check.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/platform.hh"
#include "simsycl/templates.hh"

#include <assert.h>
#include <limits>
#include <set>
#include <unordered_map>


namespace simsycl::detail {

sycl::device select_device(const device_selector &selector) {
    auto &system = simsycl::get_system_config();
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

class usm_allocation {
  public:
    usm_allocation(const sycl::context &ctx, sycl::usm::alloc kind, std::optional<sycl::device> device,
        void *const begin, void *const end)
        : m_ctx(ctx), m_kind(kind), m_device(std::move(device)), m_begin(begin), m_end(end) {
        assert(begin < end);
    }

    sycl::usm::alloc get_kind() const { return m_kind; };
    void *get_pointer() const { return m_begin; }
    size_t get_size_bytes() const { return static_cast<std::byte *>(m_end) - static_cast<std::byte *>(m_begin); }
    std::optional<sycl::context> get_context() const { return m_ctx.lock(); }
    const std::optional<sycl::device> &get_device() const { return m_device; }

  private:
    friend struct usm_allocation_order;
    weak_ref<sycl::context> m_ctx;
    sycl::usm::alloc m_kind;
    std::optional<sycl::device> m_device;
    void *m_begin, *m_end;
};

struct usm_allocation_order {
    using is_transparent = std::true_type;

    bool operator()(const usm_allocation &lhs, const usm_allocation &rhs) const {
        SIMSYCL_CHECK((lhs.m_end <= rhs.m_begin || rhs.m_end <= lhs.m_begin)
            || (lhs.m_begin == rhs.m_begin && lhs.m_end == rhs.m_end));
        return lhs.m_begin < rhs.m_begin;
    }

    bool operator()(const usm_allocation &lhs, const void *rhs) const { return lhs.m_end < rhs; }
    bool operator()(const void *lhs, const usm_allocation &rhs) const { return lhs < rhs.m_begin; }
};

struct memory_state {
    sycl::usm::alloc type;
    size_t bytes_free = 0;
    std::set<usm_allocation, usm_allocation_order> allocations;

    explicit memory_state(sycl::usm::alloc type, size_t bytes_free) : type(type), bytes_free(bytes_free) {}
};

struct system_state {
    system_config config;
    std::unordered_map<sycl::device, size_t> device_bytes_free;
    std::set<usm_allocation, usm_allocation_order> usm_allocations;

    explicit system_state(system_config config) : config(std::move(config)) {
        for(const auto &device : this->config.devices) {
            device_bytes_free.emplace(device, device.get_info<sycl::info::device::global_mem_size>());
        }
    }
};

std::optional<system_state> system;

system_state &get_system() {
    if(!detail::system.has_value()) {
        system_config config;
        auto platform = config.platforms.emplace_back(create_platform(simsycl::templates::platform::cuda_12_2));
        for(int i = 0; i < 4; ++i) {
            config.devices.push_back(create_device(platform, simsycl::templates::device::nvidia::rtx_3090));
        }
        configure_system(std::move(config));
    }
    return system.value();
}

void *usm_alloc(const sycl::context &context, sycl::usm::alloc kind, std::optional<sycl::device> device,
    size_t size_bytes, size_t alignment_bytes) {
    SIMSYCL_CHECK(kind != sycl::usm::alloc::unknown);
    SIMSYCL_CHECK((kind == sycl::usm::alloc::host) == (!device.has_value()));

    if(size_bytes == 0) { size_bytes = 1; }

    auto &system = get_system();

    size_t *bytes_free = nullptr;
    if(device.has_value()) {
        const auto context_devices = context.get_devices();
        if(std::find(context_devices.begin(), context_devices.end(), *device) == context_devices.end()) {
            throw sycl::exception(sycl::errc::invalid, "Device not associated with context");
        }

        bytes_free = &system.device_bytes_free.at(*device);
        if(*bytes_free < size_bytes) {
            throw sycl::exception(sycl::errc::memory_allocation, "Not enough memory available");
        }
    }

#if defined(_MSC_VER)
    // MSVC does not have std::aligned_alloc because the pointers it returns cannot be freed with std::free
    void *const ptr = _aligned_malloc(size_bytes, alignment_bytes);
#else
    void *const ptr = std::aligned_alloc(alignment_bytes, size_bytes);
#endif
    if(ptr == nullptr) { throw sycl::exception(sycl::errc::memory_allocation, "Not enough memory available"); }

    std::memset(ptr, static_cast<int>(uninitialized_memory_pattern), size_bytes);

    if(bytes_free != nullptr) { *bytes_free -= size_bytes; }
    system.usm_allocations.emplace(context, kind, std::move(device), ptr, static_cast<std::byte *>(ptr) + size_bytes);

    return ptr;
}

void usm_free(void *ptr, const sycl::context &context) {
    if(ptr == nullptr) return;

    auto &system = get_system();
    const auto iter = system.usm_allocations.find(ptr);
    if(iter == system.usm_allocations.end()) {
        throw sycl::exception(sycl::errc::invalid, "Pointer does not point to an allocation");
    }
    if(iter->get_pointer() != ptr) {
        throw sycl::exception(sycl::errc::invalid, "Pointer points to the inside of an allocation");
    }
    if(iter->get_context() != context) {
        throw sycl::exception(sycl::errc::invalid, "Pointer is not associated with the given context");
    }

#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif

    if(iter->get_device().has_value()) {
        system.device_bytes_free.at(iter->get_device().value()) += iter->get_size_bytes();
    }
    system.usm_allocations.erase(iter);
}

} // namespace simsycl::detail

namespace simsycl::sycl {

std::error_code make_error_code(errc e) noexcept { return {static_cast<int>(e), detail::error_category_v}; }

const std::error_category &sycl_category() noexcept { return detail::error_category_v; }

usm::alloc get_pointer_type(const void *ptr, const context &sycl_context) {
    auto &system = detail::get_system();
    if(const auto iter = system.usm_allocations.find(ptr); iter != system.usm_allocations.end()) {
        return iter->get_context() == sycl_context ? iter->get_kind() : usm::alloc::unknown;
    }
    return usm::alloc::unknown;
}

device get_pointer_device(const void *ptr, const context &sycl_context) {
    auto &system = detail::get_system();
    const auto iter = system.usm_allocations.find(ptr);
    if(iter == system.usm_allocations.end()) {
        throw sycl::exception(sycl::errc::invalid, "Pointer does not point to an allocation");
    }

    if(iter->get_kind() == usm::alloc::host) { return sycl_context.get_devices().at(0); }

    assert(iter->get_device().has_value());
    const auto &device = *iter->get_device();

    const auto context_devices = sycl_context.get_devices();
    if(std::find(context_devices.begin(), context_devices.end(), device) == context_devices.end()) {
        throw sycl::exception(sycl::errc::invalid, "Device not associated with context");
    }
    return device;
}

} // namespace simsycl::sycl

namespace simsycl {

const system_config &get_system_config() { return detail::get_system().config; }

void configure_system(system_config system) { detail::system.emplace(std::move(system)); }

} // namespace simsycl
