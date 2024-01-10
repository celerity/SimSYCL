#include <simsycl/sycl/exception.hh>
#include <simsycl/sycl/kernel.hh>

#if !defined(_MSC_VER)
// Required for kernel name demangling in Clang
#include <cxxabi.h>
#endif

namespace simsycl::detail {

const std::type_info &unnamed_kernel_type = typeid(unnamed_kernel *);

std::string demangle_name_from_pointer_type(const std::type_info &pointer_type) {
    const auto mangled = pointer_type.name();
#if !defined(_MSC_VER)
    const std::unique_ptr<char, void (*)(void *)> demangle_buffer(
        abi::__cxa_demangle(mangled, nullptr, nullptr, nullptr), std::free);
    std::string demangled = demangle_buffer != nullptr ? demangle_buffer.get() : mangled;
#else
    std::string demangled = mangled;
#endif

    // get rid of the pointer "*"
    if(!demangled.empty() && demangled.back() == '*') { demangled.pop_back(); }
    return demangled;
}

std::string get_kernel_name_string(const std::type_info &pointer_to_name_type) {
    if(pointer_to_name_type == unnamed_kernel_type) {
        return "(unnamed kernel)";
    } else {
        return demangle_name_from_pointer_type(pointer_to_name_type);
    }
}

struct kernel_id_state {
    const std::type_info &pointer_to_name_type;
    const std::type_info &func_type;
    std::string name;

    kernel_id_state(const std::type_info &pointer_to_name_type, const std::type_info &func_type)
        : pointer_to_name_type(pointer_to_name_type), func_type(func_type),
          name(get_kernel_name_string(pointer_to_name_type)) {}
};

// leaked to avoid static-destruction order issues
std::vector<sycl::kernel_id> *g_registered_kernels;

std::vector<sycl::kernel_id> &get_registered_kernels() {
    if(g_registered_kernels == nullptr) { g_registered_kernels = new std::vector<sycl::kernel_id>(); }
    return *g_registered_kernels;
}

sycl::kernel_id register_kernel(const std::type_info &pointer_to_name_type, const std::type_info &func_type) {
    const sycl::kernel_id kernel_id(pointer_to_name_type, func_type);

#if SIMSYCL_CHECK_MODE != SIMSYCL_CHECK_NONE
    for(const auto &existing_id : get_registered_kernels()) {
        const auto &existing_name_type = existing_id.state().pointer_to_name_type;
        if(existing_name_type != unnamed_kernel_type) {
            SIMSYCL_CHECK_MSG(
                existing_name_type != pointer_to_name_type, "kernel name %s not unique", kernel_id.get_name());
        }
        // the same kernel may be registered under multiple names, so we don't check for duplicate func_type
    }
#endif

    get_registered_kernels().push_back(kernel_id);
    return kernel_id;
}

sycl::kernel_id get_kernel_id(const std::type_info &pointer_to_name_type) {
    for(const auto &existing_id : get_registered_kernels()) {
        if(existing_id.state().pointer_to_name_type == pointer_to_name_type) { return existing_id; }
    }
    SIMSYCL_CHECK(false && "no such kernel in application");
    return sycl::kernel_id();
}

std::shared_ptr<kernel_bundle_state> get_kernel_bundle(const sycl::context &ctxt, const std::vector<sycl::device> &devs,
    const std::vector<sycl::kernel_id> &kernel_ids, const sycl::bundle_state state) {
    if(devs.empty()) throw sycl::exception(sycl::errc::invalid, "no devices provided to get_kernel_bundle");

    if(const auto ctxt_devs = ctxt.get_devices(); std::any_of(devs.begin(), devs.end(),
           [&](const sycl::device &d) {
               return std::find(ctxt_devs.begin(), ctxt_devs.end(), d) == ctxt_devs.end();
           })) //
    {
        throw sycl::exception(
            sycl::errc::invalid, "passed a device to get_kernel_bundle that is not part of the context");
    }

    if(state == sycl::bundle_state::input && std::any_of(devs.begin(), devs.end(), [](const sycl::device &d) {
           return !d.has(sycl::aspect::online_compiler);
       })) {
        throw sycl::exception(sycl::errc::invalid,
            "passed a device to get_kernel_bundle<bundle_state::input> that does not have aspect::online_compiler");
    }

    if(state == sycl::bundle_state::object && std::any_of(devs.begin(), devs.end(), [](const sycl::device &d) {
           return !d.has(sycl::aspect::online_linker);
       })) {
        throw sycl::exception(sycl::errc::invalid,
            "passed a device to get_kernel_bundle<bundle_state::object> that does not have aspect::online_linker");
    }

    return std::make_shared<kernel_bundle_state>(ctxt, devs, kernel_ids);
}

bool has_kernel_bundle(const sycl::context &ctxt, const std::vector<sycl::device> &devs,
    const std::vector<sycl::kernel_id> &kernel_ids, const sycl::bundle_state state) {
    if(devs.empty()) throw sycl::exception(sycl::errc::invalid, "no devices provided to has_kernel_bundle");

    if(const auto ctxt_devs = ctxt.get_devices(); std::any_of(devs.begin(), devs.end(),
           [&](const sycl::device &d) {
               return std::find(ctxt_devs.begin(), ctxt_devs.end(), d) == ctxt_devs.end();
           })) //
    {
        throw sycl::exception(
            sycl::errc::invalid, "passed a device to has_kernel_bundle that is not part of the context");
    }

    if(state == sycl::bundle_state::input && std::any_of(devs.begin(), devs.end(), [](const sycl::device &d) {
           return !d.has(sycl::aspect::online_compiler);
       })) {
        return false;
    }

    if(state == sycl::bundle_state::object && std::any_of(devs.begin(), devs.end(), [](const sycl::device &d) {
           return !d.has(sycl::aspect::online_linker);
       })) {
        return false;
    }

    return !kernel_ids.empty();
}

} // namespace simsycl::detail

namespace simsycl::sycl {

kernel_id::kernel_id(const std::type_info &kernel_name, const std::type_info &kernel_fn)
    : reference_type(std::in_place, kernel_name, kernel_fn) {}

const char *kernel_id::get_name() const noexcept { return state().name.c_str(); }

std::vector<kernel_id> get_kernel_ids() { return detail::get_registered_kernels(); }

} // namespace simsycl::sycl
