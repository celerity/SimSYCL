#include "simsycl/system.hh"
#include "simsycl/detail/allocation.hh"
#include "simsycl/detail/check.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/platform.hh"

#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <unordered_map>

#include <libenvpp/env.hpp>


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
    std::vector<sycl::platform> platforms;
    std::vector<sycl::device> devices;
    std::unordered_map<sycl::device, size_t> device_bytes_free;
    std::set<usm_allocation, usm_allocation_order> usm_allocations;

    explicit system_state(const system_config &config) {
        std::unordered_map<platform_id, sycl::platform> platforms_by_id;
        for(const auto &[id, platform_config] : config.platforms) {
            platforms_by_id.emplace(id, make_platform(platform_config));
        }
        std::unordered_map<device_id, sycl::device> devices_by_id;
        for(const auto &[id, device_config] : config.devices) {
            auto &platform = platforms_by_id.at(device_config.platform_id);
            devices_by_id.emplace(id, make_device(platform, device_config));
        }
        for(const auto &[id, device_config] : config.devices) {
            if(device_config.parent_device_id.has_value()) {
                set_parent_device(devices_by_id.at(id), devices_by_id.at(*device_config.parent_device_id));
            }
        }
        for(auto &[_, platform] : platforms_by_id) { platforms.push_back(std::move(platform)); }
        for(auto &[_, device] : devices_by_id) { devices.push_back(std::move(device)); }
    }
};

std::optional<system_state> system;

system_state &get_system() {
    if(!system.has_value()) { system.emplace(get_default_system_config()); }
    return system.value();
}

const std::vector<sycl::platform> &get_platforms() { return get_system().platforms; }
const std::vector<sycl::device> &get_devices() { return get_system().devices; }

sycl::device select_device(const device_selector &selector) {
    auto &devices = get_devices();
    SIMSYCL_CHECK(!devices.empty());
    int max_rating = std::numeric_limits<int>::lowest();
    for(const auto &device : devices) {
        if(int rating = selector(device); rating > max_rating) { max_rating = rating; }
    }
    if(max_rating < 0) { throw sycl::exception(sycl::errc::runtime, "No suitable device found"); }
    const auto device = std::find_if(
        devices.begin(), devices.end(), [&](const sycl::device &device) { return selector(device) == max_rating; });
    assert(device != devices.end());
    return *device;
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

namespace simsycl::detail {

bool g_environment_parsed = false;
std::optional<std::string> g_env_config;

void parse_environment() {
    if(g_environment_parsed) return;

    auto prefix = env::prefix("SIMSYCL");
    const auto config = prefix.register_variable<std::string>("CONFIG");
    if(const auto parsed = prefix.parse_and_validate(); parsed.ok()) {
        g_env_config = parsed.get(config);
    } else {
        std::cerr << parsed.warning_message() << parsed.error_message();
    }
    g_environment_parsed = true;
}

std::optional<system_config> g_default_system_config;

} // namespace simsycl::detail

namespace simsycl {

const system_config &get_default_system_config() {
    if(!detail::g_default_system_config.has_value()) {
        detail::parse_environment();
        if(detail::g_env_config.has_value()) {
            detail::g_default_system_config.emplace(read_system_config(*detail::g_env_config));
        } else {
            detail::g_default_system_config.emplace(builtin_system);
        }
    }
    return detail::g_default_system_config.value();
}

void configure_system(const system_config &system) { detail::system.emplace(system); }

const platform_config builtin_platform{
    .version = "0.1",
    .name = "SimSYCL",
    .vendor = "SimSYCL",
    .extensions = {},
};

const device_config builtin_device{
    .device_type = sycl::info::device_type::gpu,
    .vendor_id = 0,
    .max_compute_units = 16,
    .max_work_item_dimensions = 3,
    .max_work_item_sizes_1 = {1024},
    .max_work_item_sizes_2 = {1024, 1024},
    .max_work_item_sizes_3 = {64, 1024, 1024},
    .max_work_group_size = 1024,
    .max_num_sub_groups = 32,
    .sub_group_sizes = {32},
    .preferred_vector_width_char = 4,
    .preferred_vector_width_short = 2,
    .preferred_vector_width_int = 1,
    .preferred_vector_width_long = 1,
    .preferred_vector_width_float = 1,
    .preferred_vector_width_double = 1,
    .preferred_vector_width_half = 2,
    .native_vector_width_char = 4,
    .native_vector_width_short = 2,
    .native_vector_width_int = 1,
    .native_vector_width_long = 1,
    .native_vector_width_float = 1,
    .native_vector_width_double = 1,
    .native_vector_width_half = 2,
    .max_clock_frequency = 1000,
    .address_bits = 64,
    .max_mem_alloc_size = std::numeric_limits<std::size_t>::max(),
    .image_support = false,
    .max_read_image_args = 0,
    .max_write_image_args = 0,
    .image2d_max_height = 0,
    .image2d_max_width = 0,
    .image3d_max_height = 0,
    .image3d_max_width = 0,
    .image3d_max_depth = 0,
    .image_max_buffer_size = 0,
    .max_samplers = 0,
    .max_parameter_size = std::numeric_limits<std::size_t>::max(),
    .mem_base_addr_align = 8,
    .half_fp_config
    = {sycl::info::fp_config::denorm, sycl::info::fp_config::inf_nan, sycl::info::fp_config::round_to_nearest,
        sycl::info::fp_config::round_to_zero, sycl::info::fp_config::round_to_inf, sycl::info::fp_config::fma,
        sycl::info::fp_config::correctly_rounded_divide_sqrt},
    .single_fp_config
    = {sycl::info::fp_config::denorm, sycl::info::fp_config::inf_nan, sycl::info::fp_config::round_to_nearest,
        sycl::info::fp_config::round_to_zero, sycl::info::fp_config::round_to_inf, sycl::info::fp_config::fma,
        sycl::info::fp_config::correctly_rounded_divide_sqrt},
    .double_fp_config
    = {sycl::info::fp_config::denorm, sycl::info::fp_config::inf_nan, sycl::info::fp_config::round_to_nearest,
        sycl::info::fp_config::round_to_zero, sycl::info::fp_config::round_to_inf, sycl::info::fp_config::fma,
        sycl::info::fp_config::correctly_rounded_divide_sqrt},
    .global_mem_cache_type = sycl::info::global_mem_cache_type::read_write,
    .global_mem_cache_line_size = 128,
    .global_mem_cache_size = 16 << 20,
    .global_mem_size = std::numeric_limits<std::size_t>::max(),
    .max_constant_buffer_size = 1 << 16,
    .max_constant_args = std::numeric_limits<uint32_t>::max(),
    .local_mem_type = sycl::info::local_mem_type::local,
    .local_mem_size = 64 << 10,
    .error_correction_support = false,
    .host_unified_memory = false,
    .profiling_timer_resolution = 1,
    .is_endian_little = true,
    .is_available = true,
    .is_compiler_available = true,
    .is_linker_available = true,
    .execution_capabilities = {sycl::info::execution_capability::exec_kernel},
    .queue_profiling = true,
    .built_in_kernels = {},
    .platform_id = "SimSYCL",
    .name = "SimSYCL virtual GPU",
    .vendor = "SimSYCL",
    .driver_version = "0.1",
    .profile = "FULL_PROFILE",
    .version = "0.1",
    .aspects
    = { sycl::aspect::gpu, sycl::aspect::accelerator, sycl::aspect::fp64, sycl::aspect::atomic64,
        sycl::aspect::queue_profiling, sycl::aspect::usm_device_allocations, sycl::aspect::usm_host_allocations,
        sycl::aspect::usm_shared_allocations, },
    .extensions = {},
    .printf_buffer_size = std::numeric_limits<std::size_t>::max(),
    .preferred_interop_user_sync = true,
    .partition_max_sub_devices = 0,
    .partition_properties = {},
    .partition_affinity_domains = {sycl::info::partition_affinity_domain::not_applicable},
    .partition_type_property = sycl::info::partition_property::no_partition,
    .partition_type_affinity_domain = sycl::info::partition_affinity_domain::not_applicable,
};

const system_config builtin_system{
    .platforms = {{"SimSYCL", builtin_platform}},
    .devices = {{"GPU", builtin_device}},
};

} // namespace simsycl
