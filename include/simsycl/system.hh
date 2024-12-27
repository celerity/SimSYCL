#pragma once

#include "sycl/device.hh"
#include "sycl/platform.hh"
#include "sycl/range.hh"

#include <cstdint>
#include <vector>


namespace simsycl {

class cooperative_schedule;

/// Identifier for `sycl::platform`s within a `system_config`.
using platform_id = std::string;

/// Identifier for `sycl::device`s within a `system_config`.
using device_id = std::string;

/// Configuration for a single `sycl::device`.
///
/// All data members will be reflected in the `sycl::info::device` properties of the created device.
struct device_config {
    sycl::info::device_type device_type{};
    uint32_t vendor_id{};
    uint32_t max_compute_units{};
    uint32_t max_work_item_dimensions{};
    sycl::range<1> max_work_item_sizes_1{};
    sycl::range<2> max_work_item_sizes_2{};
    sycl::range<3> max_work_item_sizes_3{};
    size_t max_work_group_size{};
    uint32_t max_num_sub_groups{};
    std::vector<size_t> sub_group_sizes{};
    uint32_t preferred_vector_width_char{};
    uint32_t preferred_vector_width_short{};
    uint32_t preferred_vector_width_int{};
    uint32_t preferred_vector_width_long{};
    uint32_t preferred_vector_width_float{};
    uint32_t preferred_vector_width_double{};
    uint32_t preferred_vector_width_half{};
    uint32_t native_vector_width_char{};
    uint32_t native_vector_width_short{};
    uint32_t native_vector_width_int{};
    uint32_t native_vector_width_long{};
    uint32_t native_vector_width_float{};
    uint32_t native_vector_width_double{};
    uint32_t native_vector_width_half{};
    uint32_t max_clock_frequency{};
    uint32_t address_bits{};
    uint64_t max_mem_alloc_size{};
    bool image_support{};
    uint32_t max_read_image_args{};
    uint32_t max_write_image_args{};
    size_t image2d_max_height{};
    size_t image2d_max_width{};
    size_t image3d_max_height{};
    size_t image3d_max_width{};
    size_t image3d_max_depth{};
    size_t image_max_buffer_size{};
    uint32_t max_samplers{};
    size_t max_parameter_size{};
    uint32_t mem_base_addr_align{};
    std::vector<sycl::info::fp_config> half_fp_config{};
    std::vector<sycl::info::fp_config> single_fp_config{};
    std::vector<sycl::info::fp_config> double_fp_config{};
    sycl::info::global_mem_cache_type global_mem_cache_type{};
    uint32_t global_mem_cache_line_size{};
    uint64_t global_mem_cache_size{};
    uint64_t global_mem_size{};
    uint64_t max_constant_buffer_size{};
    uint32_t max_constant_args{};
    sycl::info::local_mem_type local_mem_type{};
    uint64_t local_mem_size{};
    bool error_correction_support{};
    bool host_unified_memory{};
    std::vector<sycl::memory_order> atomic_memory_order_capabilities{};
    std::vector<sycl::memory_order> atomic_fence_order_capabilities{};
    std::vector<sycl::memory_scope> atomic_memory_scope_capabilities{};
    std::vector<sycl::memory_scope> atomic_fence_scope_capabilities{};
    size_t profiling_timer_resolution{};
    bool is_endian_little{};
    bool is_available{};
    bool is_compiler_available{};
    bool is_linker_available{};
    std::vector<sycl::info::execution_capability> execution_capabilities{};
    bool queue_profiling{};
    simsycl::platform_id platform_id{};
    std::string name{};
    std::string vendor{};
    std::string driver_version{};
    std::string version{};
    std::string backend_version{};
    std::vector<sycl::aspect> aspects{};
    std::vector<std::string> extensions{};
    size_t printf_buffer_size{};
    std::optional<simsycl::device_id> parent_device_id{};
    uint32_t partition_max_sub_devices{};
    std::vector<sycl::info::partition_property> partition_properties{};
    std::vector<sycl::info::partition_affinity_domain> partition_affinity_domains{};
    sycl::info::partition_property partition_type_property{};
    sycl::info::partition_affinity_domain partition_type_affinity_domain{};
};

/// Configuration for a single `sycl::platform`.
///
/// All data members will be reflected in the `sycl::info::platform` properties of the created platform.
struct platform_config {
    std::string profile{};
    std::string version{};
    std::string name{};
    std::string vendor{};
    std::vector<std::string> extensions{};
};

/// Configuration for the entire system simulated by SimSYCL.
struct system_config {
    /// All platforms returned by `sycl::platform::get_platforms()`, in order.
    ///
    /// The `platform_id` is only relevant as a reference within this struct.
    std::unordered_map<platform_id, platform_config> platforms{};

    /// All devices returned by `sycl::device::get_devices()`, in order.
    ///
    /// The `device_id` is only relevant as a reference within this struct.
    std::unordered_map<device_id, device_config> devices{};
};

/// Configuration of the builtin platform as returned through `sycl::platform::get_platforms()` by default.
extern const platform_config builtin_platform;

/// Configuration of the builtin device as returned through `sycl::device::get_devices()` by default.
extern const device_config builtin_device;

/// Default system configuration when not overriden by the environment.
extern const system_config builtin_system;

/// Return the system configuration specified by the environment via `SIMSYCL_SYSTEM=system.json`, or `builtin_system`
/// as a fallback.
const system_config &get_default_system_config();

/// Read a `system_config` from a JSON file.
system_config read_system_config(const std::string &path_to_json_file);

/// Write a `system_config` to a JSON file.
void write_system_config(const std::string &path_to_json_file, const system_config &config);

/// Replace the active `system_config` so that all future calls to device and platform selection functions return
/// members of this configuration.
void configure_system(const system_config &system);

/// Return the schedule for threads of a kernel specified by the environment via `SIMSYCL_SCHEDULE`, or a
/// `round_robin_schedule` as a fallback.
std::shared_ptr<const cooperative_schedule> get_default_cooperative_schedule();

} // namespace simsycl

namespace simsycl::detail {

const std::vector<sycl::platform> &get_platforms(system_lock &lock);
const std::vector<sycl::device> &get_devices(system_lock &lock);
sycl::device select_device(const device_selector &selector);

} // namespace simsycl::detail
