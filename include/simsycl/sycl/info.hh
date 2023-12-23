#pragma once

#include "forward.hh"

#include <string>
#include <vector>


namespace simsycl::detail {

template<typename Result>
struct info_descriptor {
    using return_type = Result;
};

} // namespace simsycl::detail

namespace simsycl::sycl::info {

enum class device_type {
    cpu,         // Maps to OpenCL CL_DEVICE_TYPE_CPU
    gpu,         // Maps to OpenCL CL_DEVICE_TYPE_GPU
    accelerator, // Maps to OpenCL CL_DEVICE_TYPE_ACCELERATOR
    custom,      // Maps to OpenCL CL_DEVICE_TYPE_CUSTOM
    automatic,   // Maps to OpenCL CL_DEVICE_TYPE_DEFAULT
    host,
    all // Maps to OpenCL CL_DEVICE_TYPE_ALL
};

enum class event_command_status { submitted, running, complete };

enum class partition_property { no_partition, partition_equally, partition_by_counts, partition_by_affinity_domain };

enum class partition_affinity_domain {
    not_applicable,
    numa,
    L4_cache, // NOLINT(readability-identifier-naming)
    L3_cache, // NOLINT(readability-identifier-naming)
    L2_cache, // NOLINT(readability-identifier-naming)
    L1_cache, // NOLINT(readability-identifier-naming)
    next_partitionable
};

enum class local_mem_type { none, local, global };

enum class fp_config {
    denorm,
    inf_nan,
    round_to_nearest,
    round_to_zero,
    round_to_inf,
    fma,
    correctly_rounded_divide_sqrt,
    soft_float
};

enum class global_mem_cache_type { none, read_only, read_write };

enum class execution_capability { exec_kernel, exec_native_kernel };

} // namespace simsycl::sycl::info

namespace simsycl::sycl::info::context {

struct platform : detail::info_descriptor<sycl::platform> {};
struct devices : detail::info_descriptor<std::vector<sycl::device>> {};
struct atomic_memory_order_capabilities : detail::info_descriptor<std::vector<sycl::memory_order>> {};
struct atomic_fence_order_capabilities : detail::info_descriptor<std::vector<sycl::memory_order>> {};
struct atomic_memory_scope_capabilities : detail::info_descriptor<std::vector<sycl::memory_scope>> {};
struct atomic_fence_scope_capabilities : detail::info_descriptor<std::vector<sycl::memory_scope>> {};

} // namespace simsycl::sycl::info::context

namespace simsycl::sycl::info::device {

struct device_type : detail::info_descriptor<info::device_type> {};
struct vendor_id : detail::info_descriptor<uint32_t> {};
struct max_compute_units : detail::info_descriptor<uint32_t> {};
struct max_work_item_dimensions : detail::info_descriptor<uint32_t> {};
template<int Dimensions = 3>
struct max_work_item_sizes : detail::info_descriptor<range<Dimensions>> {};
struct max_work_group_size : detail::info_descriptor<size_t> {};
struct max_num_sub_groups : detail::info_descriptor<uint32_t> {};
struct sub_group_sizes : detail::info_descriptor<std::vector<size_t>> {};
struct preferred_vector_width_char : detail::info_descriptor<uint32_t> {};
struct preferred_vector_width_short : detail::info_descriptor<uint32_t> {};
struct preferred_vector_width_int : detail::info_descriptor<uint32_t> {};
struct preferred_vector_width_long : detail::info_descriptor<uint32_t> {};
struct preferred_vector_width_float : detail::info_descriptor<uint32_t> {};
struct preferred_vector_width_double : detail::info_descriptor<uint32_t> {};
struct preferred_vector_width_half : detail::info_descriptor<uint32_t> {};
struct native_vector_width_char : detail::info_descriptor<uint32_t> {};
struct native_vector_width_short : detail::info_descriptor<uint32_t> {};
struct native_vector_width_int : detail::info_descriptor<uint32_t> {};
struct native_vector_width_long : detail::info_descriptor<uint32_t> {};
struct native_vector_width_float : detail::info_descriptor<uint32_t> {};
struct native_vector_width_double : detail::info_descriptor<uint32_t> {};
struct native_vector_width_half : detail::info_descriptor<uint32_t> {};
struct max_clock_frequency : detail::info_descriptor<uint32_t> {};
struct address_bits : detail::info_descriptor<uint32_t> {};
struct max_mem_alloc_size : detail::info_descriptor<uint64_t> {};
struct [[deprecated]] image_support : detail::info_descriptor<bool> {};
struct max_read_image_args : detail::info_descriptor<uint32_t> {};
struct max_write_image_args : detail::info_descriptor<uint32_t> {};
struct image2d_max_height : detail::info_descriptor<size_t> {};
struct image2d_max_width : detail::info_descriptor<size_t> {};
struct image3d_max_height : detail::info_descriptor<size_t> {};
struct image3d_max_width : detail::info_descriptor<size_t> {};
struct image3d_max_depth : detail::info_descriptor<size_t> {};
struct image_max_buffer_size : detail::info_descriptor<size_t> {};
struct max_samplers : detail::info_descriptor<uint32_t> {};
struct max_parameter_size : detail::info_descriptor<size_t> {};
struct mem_base_addr_align : detail::info_descriptor<uint32_t> {};
struct half_fp_config : detail::info_descriptor<std::vector<info::fp_config>> {};
struct single_fp_config : detail::info_descriptor<std::vector<info::fp_config>> {};
struct double_fp_config : detail::info_descriptor<std::vector<info::fp_config>> {};
struct global_mem_cache_type : detail::info_descriptor<info::global_mem_cache_type> {};
struct global_mem_cache_line_size : detail::info_descriptor<uint32_t> {};
struct global_mem_cache_size : detail::info_descriptor<uint64_t> {};
struct global_mem_size : detail::info_descriptor<uint64_t> {};
struct [[deprecated]] max_constant_buffer_size : detail::info_descriptor<uint64_t> {};
struct [[deprecated]] max_constant_args : detail::info_descriptor<uint32_t> {};
struct local_mem_type : detail::info_descriptor<info::local_mem_type> {};
struct local_mem_size : detail::info_descriptor<uint64_t> {};
struct error_correction_support : detail::info_descriptor<bool> {};
struct host_unified_memory : detail::info_descriptor<bool> {};
struct atomic_memory_order_capabilities : detail::info_descriptor<std::vector<sycl::memory_order>> {};
struct atomic_fence_order_capabilities : detail::info_descriptor<std::vector<sycl::memory_order>> {};
struct atomic_memory_scope_capabilities : detail::info_descriptor<std::vector<sycl::memory_scope>> {};
struct atomic_fence_scope_capabilities : detail::info_descriptor<std::vector<sycl::memory_scope>> {};
struct profiling_timer_resolution : detail::info_descriptor<size_t> {};
struct is_endian_little : detail::info_descriptor<bool> {};
struct is_available : detail::info_descriptor<bool> {};
struct [[deprecated]] is_compiler_available : detail::info_descriptor<bool> {};
struct [[deprecated]] is_linker_available : detail::info_descriptor<bool> {};
struct execution_capabilities : detail::info_descriptor<std::vector<info::execution_capability>> {};
struct [[deprecated]] queue_profiling : detail::info_descriptor<bool> {};
struct [[deprecated]] built_in_kernels : detail::info_descriptor<std::vector<std::string>> {};
struct built_in_kernel_ids : detail::info_descriptor<std::vector<sycl::kernel_id>> {};
struct platform : detail::info_descriptor<sycl::platform> {};
struct name : detail::info_descriptor<std::string> {};
struct vendor : detail::info_descriptor<std::string> {};
struct driver_version : detail::info_descriptor<std::string> {};
struct profile : detail::info_descriptor<std::string> {};
struct version : detail::info_descriptor<std::string> {};
struct backend_version : detail::info_descriptor<std::string> {};
struct aspects : detail::info_descriptor<std::vector<sycl::aspect>> {};
struct [[deprecated]] extensions : detail::info_descriptor<std::vector<std::string>> {};
struct printf_buffer_size : detail::info_descriptor<size_t> {};
struct preferred_interop_user_sync : detail::info_descriptor<bool> {};
struct parent_device : detail::info_descriptor<sycl::device> {};
struct partition_max_sub_devices : detail::info_descriptor<uint32_t> {};
struct partition_properties : detail::info_descriptor<std::vector<info::partition_property>> {};
struct partition_affinity_domains : detail::info_descriptor<std::vector<info::partition_affinity_domain>> {};
struct partition_type_property : detail::info_descriptor<info::partition_property> {};
struct partition_type_affinity_domain : detail::info_descriptor<info::partition_affinity_domain> {};

} // namespace simsycl::sycl::info::device

namespace simsycl::sycl::info::event {

struct command_execution_status : detail::info_descriptor<info::event_command_status> {};

} // namespace simsycl::sycl::info::event

namespace simsycl::sycl::info::event_profiling {

struct command_submit : detail::info_descriptor<uint64_t> {};
struct command_start : detail::info_descriptor<uint64_t> {};
struct command_end : detail::info_descriptor<uint64_t> {};

} // namespace simsycl::sycl::info::event_profiling

namespace simsycl::sycl::info::kernel {

struct num_args: detail::info_descriptor<uint32_t> {};
struct attributes: detail::info_descriptor<std::string> {};

} // namespace simsycl::sycl::info::kernel

namespace simsycl::sycl::info::kernel_device_specific {

struct global_work_size : detail::info_descriptor<range<3>> {};
struct work_group_size : detail::info_descriptor<size_t> {};
struct compile_work_group_size : detail::info_descriptor<range<3>> {};
struct preferred_work_group_size_multiple : detail::info_descriptor<size_t> {};
struct private_mem_size : detail::info_descriptor<size_t> {};
struct max_num_sub_groups : detail::info_descriptor<uint32_t> {};
struct compile_num_sub_groups : detail::info_descriptor<uint32_t> {};
struct max_sub_group_size : detail::info_descriptor<uint32_t> {};
struct compile_sub_group_size : detail::info_descriptor<uint32_t> {};

} // namespace simsycl::sycl::info::kernel_device_specific

namespace simsycl::sycl::info::platform {

// info::platform::profile is undocumented - probably erroneously part of the spec Appendix A.1.
// we assume that it's equivalent to info::device::platform (at least SYCL-CTS expects a string here).
struct profile : detail::info_descriptor<std::string> {};
struct version : detail::info_descriptor<std::string> {};
struct name : detail::info_descriptor<std::string> {};
struct vendor : detail::info_descriptor<std::string> {};
struct [[deprecated]] extensions : detail::info_descriptor<std::vector<std::string>> {};

} // namespace simsycl::sycl::info::platform

namespace simsycl::sycl::info::queue {

struct context : detail::info_descriptor<sycl::context> {};
struct device : detail::info_descriptor<sycl::device> {};

} // namespace simsycl::sycl::info::queue
