#include <simsycl/system.hh>

#include <fstream>
#include <optional>

#include <nlohmann/json.hpp>


namespace nlohmann {

template<typename T>
struct adl_serializer<std::optional<T>> {
    static void to_json(json &j, const std::optional<T> &opt) {
        if(opt.has_value()) {
            j = *opt;
        } else {
            j = nullptr;
        }
    }

    static void from_json(const json &j, std::optional<T> &opt) {
        if(j.is_null()) {
            opt = std::nullopt;
        } else {
            opt = j.template get<T>();
        }
    }
};

} // namespace nlohmann

namespace simsycl::sycl { // nlohmann_json requires us to invoke SERIALIZE_ENUM in this namespace

NLOHMANN_JSON_SERIALIZE_ENUM(aspect,
    {
        {aspect::cpu, "cpu"},
        {aspect::gpu, "gpu"},
        {aspect::accelerator, "accelerator"},
        {aspect::custom, "custom"},
        {aspect::emulated, "emulated"},
        {aspect::host_debuggable, "host_debuggable"},
        {aspect::fp16, "fp16"},
        {aspect::fp64, "fp64"},
        {aspect::atomic64, "atomic64"},
        {aspect::image, "image"},
        {aspect::online_compiler, "online_compiler"},
        {aspect::online_linker, "online_linker"},
        {aspect::queue_profiling, "queue_profiling"},
        {aspect::usm_device_allocations, "usm_device_allocations"},
        {aspect::usm_host_allocations, "usm_host_allocations"},
        {aspect::usm_atomic_host_allocations, "usm_atomic_host_allocations"},
        {aspect::usm_shared_allocations, "usm_shared_allocations"},
        {aspect::usm_atomic_shared_allocations, "usm_atomic_shared_allocations"},
        {aspect::usm_system_allocations, "usm_system_allocations"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(memory_order,
    {
        {memory_order::relaxed, "relaxed"},
        {memory_order::acquire, "acquire"},
        {memory_order::release, "release"},
        {memory_order::acq_rel, "acq_rel"},
        {memory_order::seq_cst, "seq_cst"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(memory_scope,
    {
        {memory_scope::work_item, "work_item"},
        {memory_scope::sub_group, "sub_group"},
        {memory_scope::work_group, "work_group"},
        {memory_scope::device, "device"},
        {memory_scope::system, "system"},
    })

} // namespace simsycl::sycl

namespace simsycl::sycl::info { // nlohmann_json requires us to invoke SERIALIZE_ENUM in this namespace

NLOHMANN_JSON_SERIALIZE_ENUM(device_type,
    {
        {device_type::cpu, "cpu"},
        {device_type::gpu, "gpu"},
        {device_type::accelerator, "accelerator"},
        {device_type::custom, "custom"},
        {device_type::automatic, "automatic"},
        {device_type::host, "host"},
        {device_type::all, "all"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(partition_property,
    {
        {partition_property::no_partition, "no_partition"},
        {partition_property::partition_equally, "partition_equally"},
        {partition_property::partition_by_counts, "partition_by_counts"},
        {partition_property::partition_by_affinity_domain, "partition_by_affinity_domain"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(partition_affinity_domain,
    {
        {partition_affinity_domain::not_applicable, "not_applicable"},
        {partition_affinity_domain::numa, "numa"},
        {partition_affinity_domain::L4_cache, "L4_cache,"},
        {partition_affinity_domain::L3_cache, "L3_cache"},
        {partition_affinity_domain::L2_cache, "L2_cache"},
        {partition_affinity_domain::L1_cache, "L1_cache"},
        {partition_affinity_domain::next_partitionable, "next_partitionable"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(local_mem_type,
    {
        {local_mem_type::none, "none"},
        {local_mem_type::local, "local"},
        {local_mem_type::global, "global"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(fp_config,
    {
        {fp_config::denorm, "denorm"},
        {fp_config::inf_nan, "inf_nan"},
        {fp_config::round_to_nearest, "round_to_nearest"},
        {fp_config::round_to_zero, "round_to_zero"},
        {fp_config::round_to_inf, "round_to_inf"},
        {fp_config::fma, "fma"},
        {fp_config::correctly_rounded_divide_sqrt, "correctly_rounded_divide_sqrt"},
        {fp_config::soft_float, "soft_float"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(global_mem_cache_type,
    {
        {global_mem_cache_type::none, "none"},
        {global_mem_cache_type::read_only, "read_only"},
        {global_mem_cache_type::read_write, "read_write"},
    })

NLOHMANN_JSON_SERIALIZE_ENUM(execution_capability,
    {
        {execution_capability::exec_kernel, "exec_kernel"},
        {execution_capability::exec_native_kernel, "exec_native_kernel"},
    })

} // namespace simsycl::sycl::info

namespace simsycl::detail {

template<typename Interface, int Dimensions>
void to_json(nlohmann::json &json, const coordinate<Interface, Dimensions> &coord) {
    std::array<size_t, Dimensions> array;
    for(int d = 0; d < Dimensions; ++d) { array[d] = coord[d]; }
    json = nlohmann::json(array);
}

template<typename Interface, int Dimensions>
void from_json(const nlohmann::json &json, coordinate<Interface, Dimensions> &coord) {
    for(int d = 0; d < Dimensions; ++d) { coord[d] = json.at(d); }
}

} // namespace simsycl::detail

namespace simsycl {

void to_json(nlohmann::json &json, const platform_config &platform) {
    json = {
        {"profile", platform.profile},
        {"version", platform.version},
        {"name", platform.name},
        {"vendor", platform.vendor},
        {"extensions", platform.extensions},
    };
};

void from_json(const nlohmann::json &json, platform_config &platform) {
    json.at("profile").get_to(platform.profile);
    json.at("version").get_to(platform.version);
    json.at("name").get_to(platform.name);
    json.at("vendor").get_to(platform.vendor);
    json.at("extensions").get_to(platform.extensions);
};

void to_json(nlohmann::json &json, const device_config &device) {
    json = {
        {"device_type", device.device_type},
        {"vendor_id", device.vendor_id},
        {"max_compute_units", device.max_compute_units},
        {"max_work_item_dimensions", device.max_work_item_dimensions},
        {"max_work_item_sizes<1>", device.max_work_item_sizes_1},
        {"max_work_item_sizes<2>", device.max_work_item_sizes_2},
        {"max_work_item_sizes<3>", device.max_work_item_sizes_3},
        {"max_work_group_size", device.max_work_group_size},
        {"max_num_sub_groups", device.max_num_sub_groups},
        {"sub_group_sizes", device.sub_group_sizes},
        {"preferred_vector_width_char", device.preferred_vector_width_char},
        {"preferred_vector_width_short", device.preferred_vector_width_short},
        {"preferred_vector_width_int", device.preferred_vector_width_int},
        {"preferred_vector_width_long", device.preferred_vector_width_long},
        {"preferred_vector_width_float", device.preferred_vector_width_float},
        {"preferred_vector_width_double", device.preferred_vector_width_double},
        {"preferred_vector_width_half", device.preferred_vector_width_half},
        {"native_vector_width_char", device.native_vector_width_char},
        {"native_vector_width_short", device.native_vector_width_short},
        {"native_vector_width_int", device.native_vector_width_int},
        {"native_vector_width_long", device.native_vector_width_long},
        {"native_vector_width_float", device.native_vector_width_float},
        {"native_vector_width_double", device.native_vector_width_double},
        {"native_vector_width_half", device.native_vector_width_half},
        {"max_clock_frequency", device.max_clock_frequency},
        {"address_bits", device.address_bits},
        {"max_mem_alloc_size", device.max_mem_alloc_size},
        {"image_support", device.image_support},
        {"max_read_image_args", device.max_read_image_args},
        {"max_write_image_args", device.max_write_image_args},
        {"image2d_max_height", device.image2d_max_height},
        {"image2d_max_width", device.image2d_max_width},
        {"image3d_max_height", device.image3d_max_height},
        {"image3d_max_width", device.image3d_max_width},
        {"image3d_max_depth", device.image3d_max_depth},
        {"image_max_buffer_size", device.image_max_buffer_size},
        {"max_samplers", device.max_samplers},
        {"max_parameter_size", device.max_parameter_size},
        {"mem_base_addr_align", device.mem_base_addr_align},
        {"half_fp_config", device.half_fp_config},
        {"single_fp_config", device.single_fp_config},
        {"double_fp_config", device.double_fp_config},
        {"global_mem_cache_type", device.global_mem_cache_type},
        {"global_mem_cache_line_size", device.global_mem_cache_line_size},
        {"global_mem_cache_size", device.global_mem_cache_size},
        {"global_mem_size", device.global_mem_size},
        {"max_constant_buffer_size", device.max_constant_buffer_size},
        {"max_constant_args", device.max_constant_args},
        {"local_mem_type", device.local_mem_type},
        {"local_mem_size", device.local_mem_size},
        {"error_correction_support", device.error_correction_support},
        {"host_unified_memory", device.host_unified_memory},
        {"atomic_memory_order_capabilities", device.atomic_memory_order_capabilities},
        {"atomic_fence_order_capabilities", device.atomic_fence_order_capabilities},
        {"atomic_memory_scope_capabilities", device.atomic_memory_scope_capabilities},
        {"atomic_fence_scope_capabilities", device.atomic_fence_scope_capabilities},
        {"profiling_timer_resolution", device.profiling_timer_resolution},
        {"is_endian_little", device.is_endian_little},
        {"is_available", device.is_available},
        {"is_compiler_available", device.is_compiler_available},
        {"is_linker_available", device.is_linker_available},
        {"execution_capabilities", device.execution_capabilities},
        {"queue_profiling", device.queue_profiling},
        {"built_in_kernels", device.built_in_kernels},
        {"built_in_kernel_ids", device.built_in_kernel_ids},
        {"platform_id", device.platform_id},
        {"name", device.name},
        {"vendor", device.vendor},
        {"driver_version", device.driver_version},
        {"version", device.version},
        {"backend_version", device.backend_version},
        {"aspects", device.aspects},
        {"extensions", device.extensions},
        {"printf_buffer_size", device.printf_buffer_size},
        {"parent_device_id", device.parent_device_id},
        {"partition_max_sub_devices", device.partition_max_sub_devices},
        {"partition_properties", device.partition_properties},
        {"partition_affinity_domains", device.partition_affinity_domains},
        {"partition_type_property", device.partition_type_property},
        {"partition_type_affinity_domain", device.partition_type_affinity_domain},
    };
};

void from_json(const nlohmann::json &json, device_config &device) {
    json.at("device_type").get_to(device.device_type);
    json.at("vendor_id").get_to(device.vendor_id);
    json.at("max_compute_units").get_to(device.max_compute_units);
    json.at("max_work_item_dimensions").get_to(device.max_work_item_dimensions);
    json.at("max_work_item_sizes<1>").get_to(device.max_work_item_sizes_1);
    json.at("max_work_item_sizes<2>").get_to(device.max_work_item_sizes_2);
    json.at("max_work_item_sizes<3>").get_to(device.max_work_item_sizes_3);
    json.at("max_work_group_size").get_to(device.max_work_group_size);
    json.at("max_num_sub_groups").get_to(device.max_num_sub_groups);
    json.at("sub_group_sizes").get_to(device.sub_group_sizes);
    json.at("preferred_vector_width_char").get_to(device.preferred_vector_width_char);
    json.at("preferred_vector_width_short").get_to(device.preferred_vector_width_short);
    json.at("preferred_vector_width_int").get_to(device.preferred_vector_width_int);
    json.at("preferred_vector_width_long").get_to(device.preferred_vector_width_long);
    json.at("preferred_vector_width_float").get_to(device.preferred_vector_width_float);
    json.at("preferred_vector_width_double").get_to(device.preferred_vector_width_double);
    json.at("preferred_vector_width_half").get_to(device.preferred_vector_width_half);
    json.at("native_vector_width_char").get_to(device.native_vector_width_char);
    json.at("native_vector_width_short").get_to(device.native_vector_width_short);
    json.at("native_vector_width_int").get_to(device.native_vector_width_int);
    json.at("native_vector_width_long").get_to(device.native_vector_width_long);
    json.at("native_vector_width_float").get_to(device.native_vector_width_float);
    json.at("native_vector_width_double").get_to(device.native_vector_width_double);
    json.at("native_vector_width_half").get_to(device.native_vector_width_half);
    json.at("max_clock_frequency").get_to(device.max_clock_frequency);
    json.at("address_bits").get_to(device.address_bits);
    json.at("max_mem_alloc_size").get_to(device.max_mem_alloc_size);
    json.at("image_support").get_to(device.image_support);
    json.at("max_read_image_args").get_to(device.max_read_image_args);
    json.at("max_write_image_args").get_to(device.max_write_image_args);
    json.at("image2d_max_height").get_to(device.image2d_max_height);
    json.at("image2d_max_width").get_to(device.image2d_max_width);
    json.at("image3d_max_height").get_to(device.image3d_max_height);
    json.at("image3d_max_width").get_to(device.image3d_max_width);
    json.at("image3d_max_depth").get_to(device.image3d_max_depth);
    json.at("image_max_buffer_size").get_to(device.image_max_buffer_size);
    json.at("max_samplers").get_to(device.max_samplers);
    json.at("max_parameter_size").get_to(device.max_parameter_size);
    json.at("mem_base_addr_align").get_to(device.mem_base_addr_align);
    json.at("half_fp_config").get_to(device.half_fp_config);
    json.at("single_fp_config").get_to(device.single_fp_config);
    json.at("double_fp_config").get_to(device.double_fp_config);
    json.at("global_mem_cache_type").get_to(device.global_mem_cache_type);
    json.at("global_mem_cache_line_size").get_to(device.global_mem_cache_line_size);
    json.at("global_mem_cache_size").get_to(device.global_mem_cache_size);
    json.at("global_mem_size").get_to(device.global_mem_size);
    json.at("max_constant_buffer_size").get_to(device.max_constant_buffer_size);
    json.at("max_constant_args").get_to(device.max_constant_args);
    json.at("local_mem_type").get_to(device.local_mem_type);
    json.at("local_mem_size").get_to(device.local_mem_size);
    json.at("error_correction_support").get_to(device.error_correction_support);
    json.at("host_unified_memory").get_to(device.host_unified_memory);
    json.at("atomic_memory_order_capabilities").get_to(device.atomic_memory_order_capabilities);
    json.at("atomic_fence_order_capabilities").get_to(device.atomic_fence_order_capabilities);
    json.at("atomic_memory_scope_capabilities").get_to(device.atomic_memory_scope_capabilities);
    json.at("atomic_fence_scope_capabilities").get_to(device.atomic_fence_scope_capabilities);
    json.at("profiling_timer_resolution").get_to(device.profiling_timer_resolution);
    json.at("is_endian_little").get_to(device.is_endian_little);
    json.at("is_available").get_to(device.is_available);
    json.at("is_compiler_available").get_to(device.is_compiler_available);
    json.at("is_linker_available").get_to(device.is_linker_available);
    json.at("execution_capabilities").get_to(device.execution_capabilities);
    json.at("queue_profiling").get_to(device.queue_profiling);
    json.at("built_in_kernels").get_to(device.built_in_kernels);
    json.at("built_in_kernel_ids").get_to(device.built_in_kernel_ids);
    json.at("platform_id").get_to(device.platform_id);
    json.at("name").get_to(device.name);
    json.at("vendor").get_to(device.vendor);
    json.at("driver_version").get_to(device.driver_version);
    json.at("version").get_to(device.version);
    json.at("backend_version").get_to(device.backend_version);
    json.at("aspects").get_to(device.aspects);
    json.at("extensions").get_to(device.extensions);
    json.at("printf_buffer_size").get_to(device.printf_buffer_size);
    json.at("parent_device_id").get_to(device.parent_device_id);
    json.at("partition_max_sub_devices").get_to(device.partition_max_sub_devices);
    json.at("partition_properties").get_to(device.partition_properties);
    json.at("partition_affinity_domains").get_to(device.partition_affinity_domains);
    json.at("partition_type_property").get_to(device.partition_type_property);
    json.at("partition_type_affinity_domain").get_to(device.partition_type_affinity_domain);
};

void to_json(nlohmann::json &json, const system_config &system) {
    json = {
        {"platforms", system.platforms},
        {"devices", system.devices},
    };
};

void from_json(const nlohmann::json &json, system_config &system) {
    json.at("platforms").get_to(system.platforms);
    json.at("devices").get_to(system.devices);
};

system_config read_system_config(const std::string &path_to_json_file) {
    std::ifstream ifs(path_to_json_file);
    return nlohmann::json::parse(ifs).get<system_config>();
}

void write_system_config(const std::string &path_to_json_file, const system_config &config) {
    nlohmann::json json;
    to_json(json, config);
    std::ofstream(path_to_json_file) << std::setw(4) << json;
}

} // namespace simsycl
