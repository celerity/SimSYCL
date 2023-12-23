#pragma once

#include "enums.hh"
#include "forward.hh"
#include "property.hh"

#include "../detail/reference_type.hh"

#include <vector>


namespace simsycl::detail {

struct kernel_state {};
struct kernel_id_state {};
struct kernel_bundle_state {};

} // namespace simsycl::detail


namespace simsycl::sycl {

class kernel : public detail::reference_type<kernel, detail::kernel_state> {
  private:
    using reference_type = detail::reference_type<kernel, detail::kernel_state>;

  public:
    kernel() = delete;

    backend get_backend() const noexcept { return backend::simsycl; }

    context get_context() const;

    kernel_bundle<bundle_state::executable> get_kernel_bundle() const;

    template<typename Param>
    typename Param::return_type get_info() const;

    template<typename Param>
    typename Param::return_type get_info(const device &dev) const;

    template<typename Param>
    typename Param::return_type get_backend_info() const;
};

class kernel_id : public detail::reference_type<kernel_id, detail::kernel_id_state> {
  private:
    using reference_type = detail::reference_type<kernel_id, detail::kernel_id_state>;

  public:
    kernel_id() = delete;

    const char *get_name() const noexcept;
};

template<bundle_state State>
class kernel_bundle : public detail::reference_type<kernel_bundle<State>, detail::kernel_bundle_state> {
  private:
    using reference_type = detail::reference_type<kernel_bundle<State>, detail::kernel_bundle_state>;

  public:
    struct device_image_iterator;

    kernel_bundle() = delete;

    bool empty() const noexcept;

    backend get_backend() const noexcept { return backend::simsycl; }

    context get_context() const noexcept;

    std::vector<device> get_devices() const noexcept;

    bool has_kernel(const kernel_id &kernel_id) const noexcept;

    bool has_kernel(const kernel_id &kernel_id, const device &dev) const noexcept;

    template<typename KernelName>
    bool has_kernel() const noexcept;

    template<typename KernelName>
    bool has_kernel(const device &dev) const noexcept;

    std::vector<kernel_id> get_kernel_ids() const;

    kernel get_kernel(const kernel_id &kernel_id) const
        requires(State == bundle_state::executable);

    template<typename KernelName>
    kernel get_kernel() const
        requires(State == bundle_state::executable);

    bool contains_specialization_constants() const noexcept;

    bool native_specialization_constant() const noexcept;

    template<auto &SpecName>
    bool has_specialization_constant() const noexcept;

    /* Available only when:  */
    template<auto &SpecName>
    void set_specialization_constant(typename std::remove_reference_t<decltype(SpecName)>::value_type value)
        requires(State == bundle_state::input);

    template<auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant() const;

    device_image_iterator begin() const;

    device_image_iterator end() const;
};

class kernel_handler {
  public:
    template<auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant();
};

template<typename KernelName>
kernel_id get_kernel_id();

std::vector<kernel_id> get_kernel_ids();

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt);

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, const std::vector<kernel_id> &kernel_ids);

template<typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt);

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, const std::vector<device> &devs);

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(
    const context &ctxt, const std::vector<device> &devs, const std::vector<kernel_id> &kernel_ids);

template<typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, const std::vector<device> &devs);

template<bundle_state State, typename Selector>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, Selector selector);

template<bundle_state State, typename Selector>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, const std::vector<device> &devs, Selector selector);

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt);

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<kernel_id> &kernel_ids);

template<typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctxt);

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs);

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs, const std::vector<kernel_id> &kernel_ids);

template<typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs);

bool is_compatible(const std::vector<kernel_id> &kernel_ids, const device &dev);

template<typename KernelName>
bool is_compatible(const device &dev);

template<bundle_state State>
kernel_bundle<State> join(const std::vector<kernel_bundle<State>> &bundles);

kernel_bundle<bundle_state::object> compile(
    const kernel_bundle<bundle_state::input> &input_bundle, const property_list &prop_list = {});

kernel_bundle<bundle_state::object> compile(const kernel_bundle<bundle_state::input> &input_bundle,
    const std::vector<device> &devs, const property_list &prop_list = {});

kernel_bundle<bundle_state::executable> link(
    const kernel_bundle<bundle_state::object> &object_bundle, const property_list &prop_list = {});

kernel_bundle<bundle_state::executable> link(
    const std::vector<kernel_bundle<bundle_state::object>> &object_bundles, const property_list &prop_list = {});

kernel_bundle<bundle_state::executable> link(const kernel_bundle<bundle_state::object> &object_bundle,
    const std::vector<device> &devs, const property_list &prop_list = {});

kernel_bundle<bundle_state::executable> link(const std::vector<kernel_bundle<bundle_state::object>> &object_bundles,
    const std::vector<device> &devs, const property_list &prop_list = {});

kernel_bundle<bundle_state::executable> build(
    const kernel_bundle<bundle_state::input> &input_bundle, const property_list &prop_list = {});

kernel_bundle<bundle_state::executable> build(const kernel_bundle<bundle_state::input> &input_bundle,
    const std::vector<device> &devs, const property_list &prop_list = {});

} // namespace simsycl::sycl

template<>
struct std::hash<simsycl::sycl::kernel>
    : public std::hash<simsycl::detail::reference_type<simsycl::sycl::kernel, simsycl::detail::kernel_state>> {};

template<>
struct std::hash<simsycl::sycl::kernel_id>
    : public std::hash<simsycl::detail::reference_type<simsycl::sycl::kernel_id, simsycl::detail::kernel_id_state>> {};

template<simsycl::sycl::bundle_state State>
struct std::hash<simsycl::sycl::kernel_bundle<State>>
    : public std::hash<
          simsycl::detail::reference_type<simsycl::sycl::kernel_bundle<State>, simsycl::detail::kernel_bundle_state>> {
};
