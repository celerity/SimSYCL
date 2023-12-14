#include "simsycl/sycl/device.hh"
#include "simsycl/sycl/range.hh"
#include "simsycl/system.hh"


namespace simsycl::detail {

struct device_state {
    device_config config;
    weak_ref<sycl::platform, platform_state> platform;
};

int default_selector::operator()(const sycl::device &device) const {
    return device.is_gpu() || device.is_accelerator() ? 1 : 0;
}

int cpu_selector::operator()(const sycl::device &device) const { return device.is_cpu() ? 0 : -1; }

int gpu_selector ::operator()(const sycl::device &device) const { return device.is_gpu() ? 0 : -1; }

int accelerator_selector::operator()(const sycl::device &device) const { return device.is_accelerator() ? 0 : -1; }

} // namespace simsycl::detail

namespace simsycl::sycl {

device::device(detail::device_state state) : reference_type(std::in_place, std::move(state)) {}

device::device() : device(default_selector_v) {}

device::device(const detail::device_selector &device_selector) : device(detail::select_device(device_selector)) {}

template<>
info::device_type device::get_info<info::device::device_type>() const {
    return state().config.device_type;
}

template<>
uint32_t device::get_info<info::device::vendor_id>() const {
    return state().config.vendor_id;
}

template<>
uint32_t device::get_info<info::device::max_compute_units>() const {
    return state().config.max_compute_units;
}

template<>
uint32_t device::get_info<info::device::max_work_item_dimensions>() const {
    return state().config.max_work_item_dimensions;
}

template<>
range<1> device::get_info<info::device::max_work_item_sizes<1>>() const {
    return state().config.max_work_item_sizes_1;
}

template<>
range<2> device::get_info<info::device::max_work_item_sizes<2>>() const {
    return state().config.max_work_item_sizes_2;
}

template<>
range<3> device::get_info<info::device::max_work_item_sizes<3>>() const {
    return state().config.max_work_item_sizes_3;
}

template<>
size_t device::get_info<info::device::max_work_group_size>() const {
    return state().config.max_work_group_size;
}

template<>
uint32_t device::get_info<info::device::max_num_sub_groups>() const {
    return state().config.max_num_sub_groups;
}

template<>
std::vector<size_t> device::get_info<info::device::sub_group_sizes>() const {
    return state().config.sub_group_sizes;
}

template<>
uint32_t device::get_info<info::device::preferred_vector_width_char>() const {
    return state().config.preferred_vector_width_char;
}

template<>
uint32_t device::get_info<info::device::preferred_vector_width_short>() const {
    return state().config.preferred_vector_width_short;
}

template<>
uint32_t device::get_info<info::device::preferred_vector_width_int>() const {
    return state().config.preferred_vector_width_int;
}

template<>
uint32_t device::get_info<info::device::preferred_vector_width_long>() const {
    return state().config.preferred_vector_width_long;
}

template<>
uint32_t device::get_info<info::device::preferred_vector_width_float>() const {
    return state().config.preferred_vector_width_float;
}

template<>
uint32_t device::get_info<info::device::preferred_vector_width_double>() const {
    return state().config.preferred_vector_width_double;
}

template<>
uint32_t device::get_info<info::device::preferred_vector_width_half>() const {
    return state().config.preferred_vector_width_half;
}

template<>
uint32_t device::get_info<info::device::native_vector_width_char>() const {
    return state().config.native_vector_width_char;
}

template<>
uint32_t device::get_info<info::device::native_vector_width_short>() const {
    return state().config.native_vector_width_short;
}

template<>
uint32_t device::get_info<info::device::native_vector_width_int>() const {
    return state().config.native_vector_width_int;
}

template<>
uint32_t device::get_info<info::device::native_vector_width_long>() const {
    return state().config.native_vector_width_long;
}

template<>
uint32_t device::get_info<info::device::native_vector_width_float>() const {
    return state().config.native_vector_width_float;
}
template<>
uint32_t device::get_info<info::device::native_vector_width_double>() const {
    return state().config.native_vector_width_double;
}

template<>
uint32_t device::get_info<info::device::native_vector_width_half>() const {
    return state().config.native_vector_width_half;
}
template<>
uint32_t device::get_info<info::device::max_clock_frequency>() const {
    return state().config.max_clock_frequency;
}

template<>
uint32_t device::get_info<info::device::address_bits>() const {
    return state().config.address_bits;
}

template<>
uint64_t device::get_info<info::device::max_mem_alloc_size>() const {
    return state().config.max_mem_alloc_size;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
bool device::get_info<info::device::image_support>() const {
    return state().config.image_support;
}
#pragma GCC diagnostic pop

template<>
uint32_t device::get_info<info::device::max_read_image_args>() const {
    return state().config.max_read_image_args;
}

template<>
uint32_t device::get_info<info::device::max_write_image_args>() const {
    return state().config.max_write_image_args;
}

template<>
size_t device::get_info<info::device::image2d_max_height>() const {
    return state().config.image2d_max_height;
}

template<>
size_t device::get_info<info::device::image2d_max_width>() const {
    return state().config.image2d_max_width;
}
template<>
size_t device::get_info<info::device::image3d_max_height>() const {
    return state().config.image3d_max_height;
}

template<>
size_t device::get_info<info::device::image3d_max_width>() const {
    return state().config.image3d_max_width;
}

template<>
size_t device::get_info<info::device::image3d_max_depth>() const {
    return state().config.image3d_max_depth;
}

template<>
size_t device::get_info<info::device::image_max_buffer_size>() const {
    return state().config.image_max_buffer_size;
}

template<>
uint32_t device::get_info<info::device::max_samplers>() const {
    return state().config.max_samplers;
}

template<>
size_t device::get_info<info::device::max_parameter_size>() const {
    return state().config.max_parameter_size;
}

template<>
uint32_t device::get_info<info::device::mem_base_addr_align>() const {
    return state().config.mem_base_addr_align;
}

template<>
std::vector<info::fp_config> device::get_info<info::device::half_fp_config>() const {
    return state().config.half_fp_config;
}

template<>
std::vector<info::fp_config> device::get_info<info::device::single_fp_config>() const {
    return state().config.single_fp_config;
}

template<>
std::vector<info::fp_config> device::get_info<info::device::double_fp_config>() const {
    return state().config.double_fp_config;
}

template<>
info::global_mem_cache_type device::get_info<info::device::global_mem_cache_type>() const {
    return state().config.global_mem_cache_type;
}

template<>
uint32_t device::get_info<info::device::global_mem_cache_line_size>() const {
    return state().config.global_mem_cache_line_size;
}

template<>
uint64_t device::get_info<info::device::global_mem_cache_size>() const {
    return state().config.global_mem_cache_size;
}

template<>
uint64_t device::get_info<info::device::global_mem_size>() const {
    return state().config.global_mem_size;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
uint64_t device::get_info<info::device::max_constant_buffer_size>() const {
    return state().config.max_constant_buffer_size;
}
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
uint32_t device::get_info<info::device::max_constant_args>() const {
    return state().config.max_constant_args;
}
#pragma GCC diagnostic pop

template<>
info::local_mem_type device::get_info<info::device::local_mem_type>() const {
    return state().config.local_mem_type;
}

template<>
uint64_t device::get_info<info::device::local_mem_size>() const {
    return state().config.local_mem_size;
}

template<>
bool device::get_info<info::device::error_correction_support>() const {
    return state().config.error_correction_support;
}

template<>
bool device::get_info<info::device::host_unified_memory>() const {
    return state().config.host_unified_memory;
}

template<>
std::vector<sycl::memory_order> device::get_info<info::device::atomic_memory_order_capabilities>() const {
    return state().config.atomic_memory_order_capabilities;
}

template<>
std::vector<sycl::memory_order> device::get_info<info::device::atomic_fence_order_capabilities>() const {
    return state().config.atomic_fence_order_capabilities;
}

template<>
std::vector<sycl::memory_scope> device::get_info<info::device::atomic_memory_scope_capabilities>() const {
    return state().config.atomic_memory_scope_capabilities;
}

template<>
std::vector<sycl::memory_scope> device::get_info<info::device::atomic_fence_scope_capabilities>() const {
    return state().config.atomic_fence_scope_capabilities;
}

template<>
size_t device::get_info<info::device::profiling_timer_resolution>() const {
    return state().config.profiling_timer_resolution;
}

template<>
bool device::get_info<info::device::is_endian_little>() const {
    return state().config.is_endian_little;
}

template<>
bool device::get_info<info::device::is_available>() const {
    return state().config.is_available;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
bool device::get_info<info::device::is_compiler_available>() const {
    return state().config.is_compiler_available;
}
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
bool device::get_info<info::device::is_linker_available>() const {
    return state().config.is_linker_available;
}
#pragma GCC diagnostic pop

template<>
std::vector<info::execution_capability> device::get_info<info::device::execution_capabilities>() const {
    return state().config.execution_capabilities;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
bool device::get_info<info::device::queue_profiling>() const {
    return state().config.queue_profiling;
}
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
std::vector<std::string> device::get_info<info::device::built_in_kernels>() const {
    return state().config.built_in_kernels;
}
#pragma GCC diagnostic pop

template<>
std::vector<sycl::kernel_id> device::get_info<info::device::built_in_kernel_ids>() const {
    return state().config.built_in_kernel_ids;
}

template<>
sycl::platform device::get_info<info::device::platform>() const {
    return state().platform.lock();
}

template<>
std::string device::get_info<info::device::name>() const {
    return state().config.name;
}

template<>
std::string device::get_info<info::device::vendor>() const {
    return state().config.vendor;
}

template<>
std::string device::get_info<info::device::driver_version>() const {
    return state().config.driver_version;
}

template<>
std::string device::get_info<info::device::profile>() const {
    return state().config.profile;
}

template<>
std::string device::get_info<info::device::version>() const {
    return state().config.version;
}

template<>
std::string device::get_info<info::device::backend_version>() const {
    return state().config.backend_version;
}

template<>
std::vector<sycl::aspect> device::get_info<info::device::aspects>() const {
    return state().config.aspects;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template<>
std::vector<std::string> device::get_info<info::device::extensions>() const {
    return state().config.extensions;
}
#pragma GCC diagnostic pop

template<>
size_t device::get_info<info::device::printf_buffer_size>() const {
    return state().config.printf_buffer_size;
}

template<>
bool device::get_info<info::device::preferred_interop_user_sync>() const {
    return state().config.preferred_interop_user_sync;
}

template<>
sycl::device device::get_info<info::device::parent_device>() const {
    return state().config.parent_device.value();
}

template<>
uint32_t device::get_info<info::device::partition_max_sub_devices>() const {
    return state().config.partition_max_sub_devices;
}

template<>
std::vector<info::partition_property> device::get_info<info::device::partition_properties>() const {
    return state().config.partition_properties;
}

template<>
std::vector<info::partition_affinity_domain> device::get_info<info::device::partition_affinity_domains>() const {
    return state().config.partition_affinity_domains;
}

template<>
info::partition_property device::get_info<info::device::partition_type_property>() const {
    return state().config.partition_type_property;
}

template<>
info::partition_affinity_domain device::get_info<info::device::partition_type_affinity_domain>() const {
    return state().config.partition_type_affinity_domain;
}

platform device::get_platform() const { return get_info<info::device::platform>(); }

bool device::has(aspect asp) const {
    return std::find(state().config.aspects.begin(), state().config.aspects.end(), asp) != state().config.aspects.end();
}

[[deprecated]] bool device::has_extension(const std::string &extension) const {
    return std::find(state().config.extensions.begin(), state().config.extensions.end(), extension)
        != state().config.extensions.end();
}

std::vector<device> device::get_devices(info::device_type type) {
    auto &system = get_system();
    std::vector<device> result;
    std::copy_if(system.devices.begin(), system.devices.end(), std::back_inserter(result),
        [type](const device &dev) { return dev.get_info<info::device::device_type>() == type; });

    return result;
}

} // namespace simsycl::sycl

namespace simsycl {

sycl::device create_device(sycl::platform &platform, const device_config &config) {
    detail::device_state state;
    state.config = config;
    state.platform = platform.weak_ref();
    sycl::device device(state);
    platform.add_device(device);
    return device;
}

} // namespace simsycl
