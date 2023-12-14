#include "simsycl/templates.hh"
#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/platform.hh"
#include "simsycl/system.hh"


namespace simsycl::templates::platform {

const platform_config cuda_12_2{
    .version = "12.2.0",
    .name = "CUDA",
    .vendor = "NVIDIA",
    .extensions = {},
};

}

namespace simsycl::templates::device::nvidia {

const device_config rtx_3090{
    .device_type = sycl::info::device_type::gpu,
    .vendor_id = 4318,
    .max_compute_units = 82,
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
    .max_clock_frequency = 1695,
    .address_bits = 64,
    .max_mem_alloc_size = 25438126080ull,
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
    .max_parameter_size = 18446744073709551615ull,
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
    .global_mem_cache_size = 6291456,
    .global_mem_size = 25438126080,
    .max_constant_buffer_size = 65536,
    .max_constant_args = 4294967295,
    .local_mem_type = sycl::info::local_mem_type::local,
    .local_mem_size = 49152,
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
    .name = "NVIDIA GeForce RTX 3090",
    .vendor = "NVIDIA",
    .driver_version = "12010",
    .profile = "FULL_PROFILE",
    .version = "sm_86",
    .aspects
    = { sycl::aspect::gpu, sycl::aspect::accelerator, sycl::aspect::fp64, sycl::aspect::atomic64,
        sycl::aspect::queue_profiling, sycl::aspect::usm_device_allocations, sycl::aspect::usm_host_allocations,
        sycl::aspect::usm_shared_allocations, },
    .extensions = {},
    .printf_buffer_size = 18446744073709551615ull,
    .preferred_interop_user_sync = true,
    .partition_max_sub_devices = 0,
    .partition_properties = {},
    .partition_affinity_domains = {sycl::info::partition_affinity_domain::not_applicable},
    .partition_type_property = sycl::info::partition_property::no_partition,
    .partition_type_affinity_domain = sycl::info::partition_affinity_domain::not_applicable,
};

} // namespace simsycl::templates::device::nvidia
