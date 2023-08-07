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

namespace simsycl::sycl::usm {

enum class alloc { host, device, shared, unknown };

}
