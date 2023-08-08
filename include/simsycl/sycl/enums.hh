#pragma once

namespace simsycl::sycl {

enum class addressing_mode { mirrored_repeat, repeat, clamp_to_edge, clamp, none };

enum class filtering_mode { nearest, linear };

enum class coordinate_normalization_mode { normalized, unnormalized };

struct image_sampler {
    addressing_mode addressing;
    coordinate_normalization_mode coordinate;
    filtering_mode filtering;
};

enum class access_mode {
    read,
    write,
    read_write,
    discard_write,      // Deprecated in SYCL 2020
    discard_read_write, // Deprecated in SYCL 2020
    atomic              // Deprecated in SYCL 2020
};

enum class aspect {
    cpu,
    gpu,
    accelerator,
    custom,
    emulated,
    host_debuggable,
    fp16,
    fp64,
    atomic64,
    image,
    online_compiler,
    online_linker,
    queue_profiling,
    usm_device_allocations,
    usm_host_allocations,
    usm_atomic_host_allocations,
    usm_shared_allocations,
    usm_atomic_shared_allocations,
    usm_system_allocations
};

enum class backend {};

enum class bundle_state { input, object, executable };

enum class errc {
    success = 0,
    runtime,
    kernel,
    accessor,
    nd_range,
    event,
    kernel_argument,
    build,
    invalid,
    memory_allocation,
    platform,
    profiling,
    feature_not_supported,
    kernel_not_supported,
    backend_mismatch
};

enum class image_format {
    r8g8b8a8_unorm,
    r16g16b16a16_unorm,
    r8g8b8a8_sint,
    r16g16b16a16_sint,
    r32b32g32a32_sint,
    r8g8b8a8_uint,
    r16g16b16a16_uint,
    r32b32g32a32_uint,
    r16b16g16a16_sfloat,
    r32g32b32a32_sfloat,
    b8g8r8a8_unorm
};

enum class image_target { device, host_task };

enum class memory_order { relaxed, acquire, release, acq_rel, seq_cst };

inline constexpr auto memory_order_relaxed = memory_order::relaxed;
inline constexpr auto memory_order_acquire = memory_order::acquire;
inline constexpr auto memory_order_release = memory_order::release;
inline constexpr auto memory_order_acq_rel = memory_order::acq_rel;
inline constexpr auto memory_order_seq_cst = memory_order::seq_cst;

enum class memory_scope { work_item, sub_group, work_group, device, system };

inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;
inline constexpr auto memory_scope_system = memory_scope::system;

enum class rounding_mode { automatic, rte, rtz, rtp, rtn };

enum class stream_manipulator {
    flush,
    dec,
    hex,
    oct,
    noshowbase,
    showbase,
    noshowpos,
    showpos,
    endl,
    fixed,
    scientific,
    hexfloat,
    defaultfloat
};

enum class target {
    device,
    host_task,
    constant_buffer,       // Deprecated
    local,                 // Deprecated
    host_buffer,           // Deprecated
    global_buffer = device // Deprecated
};

} // namespace simsycl::sycl

namespace simsycl::sycl::access {

enum class address_space {
    global_space,
    local_space,
    constant_space, // Deprecated in SYCL 2020
    private_space,
    generic_space
};

enum class decorated { no, yes, legacy };

// The legacy type "access::mode" is deprecated.
using mode = sycl::access_mode;

// The legacy type "access::target" is deprecated.
using sycl::target;

enum class placeholder { // Deprecated
    false_t,
    true_t
};

} // namespace simsycl::sycl::access

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

struct platform;
struct devices;
struct atomic_memory_order_capabilities;
struct atomic_fence_order_capabilities;
struct atomic_memory_scope_capabilities;
struct atomic_fence_scope_capabilities;

} // namespace simsycl::sycl::info::context

namespace simsycl::sycl::info::device {

struct device_type;
struct vendor_id;
struct max_compute_units;
struct max_work_item_dimensions;
template <int Dimensions = 3>
struct max_work_item_sizes;
struct max_work_group_size;
struct preferred_vector_width_char;
struct preferred_vector_width_short;
struct preferred_vector_width_int;
struct preferred_vector_width_long;
struct preferred_vector_width_float;
struct preferred_vector_width_double;
struct preferred_vector_width_half;
struct native_vector_width_char;
struct native_vector_width_short;
struct native_vector_width_int;
struct native_vector_width_long;
struct native_vector_width_float;
struct native_vector_width_double;
struct native_vector_width_half;
struct max_clock_frequency;
struct address_bits;
struct max_mem_alloc_size;
struct [[deprecated]] image_support;
struct max_read_image_args;
struct max_write_image_args;
struct image2d_max_height;
struct image2d_max_width;
struct image3d_max_height;
struct image3d_max_width;
struct image3d_max_depth;
struct image_max_buffer_size;
struct max_samplers;
struct max_parameter_size;
struct mem_base_addr_align;
struct half_fp_config;
struct single_fp_config;
struct double_fp_config;
struct global_mem_cache_type;
struct global_mem_cache_line_size;
struct global_mem_cache_size;
struct global_mem_size;
struct [[deprecated]] max_constant_buffer_size;
struct [[deprecated]] max_constant_args;
struct local_mem_type;
struct local_mem_size;
struct error_correction_support;
struct host_unified_memory;
struct atomic_memory_order_capabilities;
struct atomic_fence_order_capabilities;
struct atomic_memory_scope_capabilities;
struct atomic_fence_scope_capabilities;
struct profiling_timer_resolution;
struct is_endian_little;
struct is_available;
struct [[deprecated]] is_compiler_available;
struct [[deprecated]] is_linker_available;
struct execution_capabilities;
struct [[deprecated]] queue_profiling;
struct [[deprecated]] built_in_kernels;
struct built_in_kernel_ids;
struct platform;
struct name;
struct vendor;
struct driver_version;
struct profile;
struct version;
struct backend_version;
struct aspects;
struct [[deprecated]] extensions;
struct printf_buffer_size;
struct preferred_interop_user_sync;
struct parent_device;
struct partition_max_sub_devices;
struct partition_properties;
struct partition_affinity_domains;
struct partition_type_property;
struct partition_type_affinity_domain;

} // namespace simsycl::sycl::info::device

namespace simsycl::sycl::info::event {

struct command_execution_status;

} // namespace simsycl::sycl::info::event

namespace simsycl::sycl::info::event_profiling {

struct command_submit;
struct command_start;
struct command_end;

} // namespace simsycl::sycl::info::event_profiling

namespace simsycl::sycl::info::kernel {

struct num_args;
struct attributes;

} // namespace simsycl::sycl::info::kernel

namespace simsycl::sycl::info::kernel_device_specific {

struct global_work_size;
struct work_group_size;
struct compile_work_group_size;
struct preferred_work_group_size_multiple;
struct private_mem_size;
struct max_num_sub_groups;
struct compile_num_sub_groups;
struct max_sub_group_size;
struct compile_sub_group_size;

} // namespace simsycl::sycl::info::kernel_device_specific

namespace simsycl::sycl::info::platform {

struct profile;
struct version;
struct name;
struct vendor;
struct [[deprecated]] extensions;

} // namespace simsycl::sycl::info::platform

namespace simsycl::sycl::info::queue {

struct context;
struct device;

} // namespace simsycl::sycl::info::queue

namespace simsycl::sycl::usm {

enum class alloc { host, device, shared, unknown };

}
