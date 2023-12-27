#pragma once

#include "detail/check.hh"

#include "sycl/device.hh"
#include "sycl/exception.hh"
#include "sycl/kernel.hh"
#include "sycl/platform.hh"
#include "sycl/range.hh"

#include <vector>


namespace simsycl {

using platform_id = std::string;
using device_id = std::string;

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
    std::vector<std::string> built_in_kernels{};
    std::vector<std::string> built_in_kernel_ids{};
    simsycl::platform_id platform_id{};
    std::string name{};
    std::string vendor{};
    std::string driver_version{};
    std::string profile{};
    std::string version{};
    std::string backend_version{};
    std::vector<sycl::aspect> aspects{};
    std::vector<std::string> extensions{};
    size_t printf_buffer_size{};
    bool preferred_interop_user_sync{};
    std::optional<simsycl::device_id> parent_device_id{};
    uint32_t partition_max_sub_devices{};
    std::vector<sycl::info::partition_property> partition_properties{};
    std::vector<sycl::info::partition_affinity_domain> partition_affinity_domains{};
    sycl::info::partition_property partition_type_property{};
    sycl::info::partition_affinity_domain partition_type_affinity_domain{};
};

struct platform_config {
    std::string profile{};
    std::string version{};
    std::string name{};
    std::string vendor{};
    std::vector<std::string> extensions{};
};

struct system_config {
    std::unordered_map<platform_id, platform_config> platforms{};
    std::unordered_map<device_id, device_config> devices{};
};

void configure_system(const system_config &system);

} // namespace simsycl

namespace simsycl::detail {

const std::vector<sycl::platform> &get_platforms();
const std::vector<sycl::device> &get_devices();
sycl::device select_device(const device_selector &selector);

} // namespace simsycl::detail
