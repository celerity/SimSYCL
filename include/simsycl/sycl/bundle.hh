#pragma once

#include "enums.hh"
#include "forward.hh"
#include "property.hh"

#include <vector>


namespace simsycl::sycl {

class kernel_id { /* ... */
};

template<bundle_state State>
class kernel_bundle { /* ... */
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
