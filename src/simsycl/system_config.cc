#include <simsycl/system.hh>

#include <nlohmann/json.hpp>


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

}

namespace simsycl::detail {

device_config load_device_config(const std::string &path_to_json_file) {}

} // namespace simsycl::detail