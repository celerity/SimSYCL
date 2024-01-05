#pragma once

#include "context.hh"
#include "device.hh"
#include "enums.hh"
#include "forward.hh"
#include "property.hh"

#include "../detail/reference_type.hh"

#include <typeinfo>
#include <vector>


namespace simsycl::detail {

template<typename T>
struct is_specialization_id : std::false_type {};

template<typename T>
struct is_specialization_id<sycl::specialization_id<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_specialization_id_v = is_specialization_id<T>::value;

template<typename T>
const T &get_specialization_default(const sycl::specialization_id<T> &id) {
    return id.m_default_value;
}

} // namespace simsycl::detail

namespace simsycl::sycl {

template<typename T>
class specialization_id {
  public:
    using value_type = T;

    template<class... Args>
    explicit constexpr specialization_id(Args &&...args)
        requires(std::is_constructible_v<T, Args...>)
        : m_default_value(std::forward<Args>(args)...) {}

    specialization_id(const specialization_id &rhs) = delete;
    specialization_id(specialization_id &&rhs) = delete;
    specialization_id &operator=(const specialization_id &rhs) = delete;
    specialization_id &operator=(specialization_id &&rhs) = delete;

  private:
    template<typename U>
    friend const U &detail::get_specialization_default(const sycl::specialization_id<U> &id);

    value_type m_default_value;
};

class kernel_handler {
  public:
    template<auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant();
    // implemented in handler.hh

  private:
    friend class handler;
    explicit kernel_handler(handler *cgh) : m_cgh(cgh) {}

    handler *m_cgh;
};

} // namespace simsycl::sycl

namespace simsycl::detail {

struct kernel_id_state {
    const std::type_info &pointer_to_name_type;
    const std::type_info &func_type;
    std::string name;

    kernel_id_state(const std::type_info &pointer_to_name_type, const std::type_info &func_type);
};

sycl::kernel_id register_kernel(const std::type_info &pointer_to_name_type, const std::type_info &func_type);
sycl::kernel_id get_kernel_id(const std::type_info &pointer_to_name_type);

} // namespace simsycl::detail

namespace simsycl::sycl {

class kernel_id : public detail::reference_type<kernel_id, detail::kernel_id_state> {
  private:
    using reference_type = detail::reference_type<kernel_id, detail::kernel_id_state>;

  public:
    const char *get_name() const noexcept { return state().name.c_str(); }

  private:
    kernel_id() = default;

    friend kernel_id detail::register_kernel(
        const std::type_info &pointer_to_name_type, const std::type_info &func_type);
    friend kernel_id detail::get_kernel_id(const std::type_info &pointer_to_name_type);

    explicit kernel_id(const std::type_info &kernel_name, const std::type_info &kernel_fn)
        : reference_type(std::in_place, kernel_name, kernel_fn) {}
};

} // namespace simsycl::sycl

namespace simsycl::detail {

struct kernel_bundle_state {
    sycl::context context;
    std::vector<sycl::device> devices;
    std::vector<sycl::kernel_id> kernel_ids;

    kernel_bundle_state(const sycl::context &context, const std::vector<sycl::device> &devices,
        const std::vector<sycl::kernel_id> &kernel_ids)
        : context(context), devices(devices), kernel_ids(kernel_ids) {}
};

template<sycl::bundle_state State>
sycl::kernel_bundle<State> make_kernel_bundle(std::shared_ptr<detail::kernel_bundle_state> state);

template<sycl::bundle_state NewState, sycl::bundle_state OldState>
sycl::kernel_bundle<NewState> kernel_bundle_cast(const sycl::kernel_bundle<OldState> &bundle);

} // namespace simsycl::detail

namespace simsycl::sycl {

template<bundle_state State>
class kernel_bundle : public detail::reference_type<kernel_bundle<State>, detail::kernel_bundle_state> {
  private:
    using reference_type = detail::reference_type<kernel_bundle<State>, detail::kernel_bundle_state>;
    using reference_type::state;

  public:
    struct device_image_iterator; // TODO

    kernel_bundle() = delete;

    bool empty() const noexcept { return state().kernel_ids.empty(); }

    backend get_backend() const noexcept { return backend::simsycl; }

    context get_context() const noexcept { return state().context; }

    std::vector<device> get_devices() const noexcept { return state().devices; }

    bool has_kernel(const kernel_id &kernel_id) const noexcept {
        return std::find(state().kernel_ids.begin(), state().kernel_ids.end(), kernel_id) != state().kernel_ids.end();
    }

    bool has_kernel(const kernel_id &kernel_id, const device &dev) const noexcept {
        (void)dev;
        return has_kernel(kernel_id);
    }

    template<typename KernelName>
    bool has_kernel() const noexcept {
        return has_kernel(get_kernel_id<KernelName>());
    }

    template<typename KernelName>
    bool has_kernel(const device &dev) const noexcept {
        return has_kernel(get_kernel_id<KernelName>(), dev);
    }

    std::vector<kernel_id> get_kernel_ids() const { return state().kernel_ids; }

    kernel get_kernel(const kernel_id &kernel_id) const
        requires(State == bundle_state::executable);

    template<typename KernelName>
    kernel get_kernel() const
        requires(State == bundle_state::executable);

    bool contains_specialization_constants() const noexcept; // TODO

    bool native_specialization_constant() const noexcept; // TODO

    template<auto &SpecName>
    bool has_specialization_constant() const noexcept; // TODO

    /* Available only when:  */
    template<auto &SpecName>
    void set_specialization_constant(typename std::remove_reference_t<decltype(SpecName)>::value_type value)
        requires(State == bundle_state::input); // TODO

    template<auto &SpecName>
    typename std::remove_reference_t<decltype(SpecName)>::value_type get_specialization_constant() const; // TODO

    device_image_iterator begin() const;

    device_image_iterator end() const;

  private:
    template<bundle_state>
    friend class kernel_bundle;

    template<sycl::bundle_state S>
    friend kernel_bundle<S> detail::make_kernel_bundle(std::shared_ptr<detail::kernel_bundle_state> state);

    template<sycl::bundle_state NewState, sycl::bundle_state OldState>
    friend kernel_bundle<NewState> detail::kernel_bundle_cast(const kernel_bundle<OldState> &bundle);

    explicit kernel_bundle(std::shared_ptr<detail::kernel_bundle_state> &&state) : reference_type(std::move(state)) {}
};

} // namespace simsycl::sycl

namespace simsycl::detail {

template<sycl::bundle_state State>
sycl::kernel_bundle<State> make_kernel_bundle(std::shared_ptr<kernel_bundle_state> state) {
    return sycl::kernel_bundle<State>(std::move(state));
}

template<sycl::bundle_state NewState, sycl::bundle_state OldState>
sycl::kernel_bundle<NewState> kernel_bundle_cast(const sycl::kernel_bundle<OldState> &bundle) {
    return sycl::kernel_bundle<NewState>(bundle.state());
}

struct kernel_state {
    sycl::kernel_bundle<sycl::bundle_state::executable> bundle;
    sycl::kernel_id id;

    kernel_state(const sycl::kernel_bundle<sycl::bundle_state::executable> &bundle, const sycl::kernel_id &id)
        : bundle(bundle), id(id) {}
};

} // namespace simsycl::detail

namespace simsycl::sycl {

class kernel : public detail::reference_type<kernel, detail::kernel_state> {
  private:
    using reference_type = detail::reference_type<kernel, detail::kernel_state>;

  public:
    kernel() = delete;

    backend get_backend() const noexcept { return backend::simsycl; }

    context get_context() const { return state().bundle.get_context(); }

    kernel_bundle<bundle_state::executable> get_kernel_bundle() const { return state().bundle; }

    template<typename Param>
    typename Param::return_type get_info() const;

    template<typename Param>
    typename Param::return_type get_info(const device &dev) const;

    template<typename Param>
    typename Param::return_type get_backend_info() const;

  private:
    template<bundle_state State>
    friend class kernel_bundle;

    explicit kernel(const kernel_bundle<bundle_state::executable> &bundle, const kernel_id &id)
        : reference_type(std::in_place, bundle, id) {}
};

} // namespace simsycl::sycl

namespace simsycl::sycl {

template<bundle_state State>
kernel kernel_bundle<State>::get_kernel(const kernel_id &kernel_id) const
    requires(State == bundle_state::executable)
{
    SIMSYCL_CHECK(has_kernel(kernel_id));
    return kernel(*this, kernel_id);
}

template<bundle_state State>
template<typename KernelName>
kernel kernel_bundle<State>::get_kernel() const
    requires(State == bundle_state::executable)
{
    return get_kernel(get_kernel_id<KernelName>());
}

template<typename KernelName>
kernel_id get_kernel_id() {
    return detail::get_kernel_id(typeid(KernelName *));
}

std::vector<kernel_id> get_kernel_ids();

} // namespace simsycl::sycl

namespace simsycl::detail {

std::shared_ptr<kernel_bundle_state> get_kernel_bundle(const sycl::context &ctxt, const std::vector<sycl::device> &devs,
    const std::vector<sycl::kernel_id> &kernel_ids, const sycl::bundle_state state);

bool has_kernel_bundle(const sycl::context &ctxt, const std::vector<sycl::device> &devs,
    const std::vector<sycl::kernel_id> &kernel_ids, const sycl::bundle_state state);

struct device_image_state {};

} // namespace simsycl::detail

namespace simsycl::sycl {

template<bundle_state State>
class device_image final : public detail::reference_type<device_image<State>, detail::device_image_state> {
  private:
    using reference_type = detail::reference_type<device_image<State>, detail::device_image_state>;

  public:
    device_image() = delete;

    bool has_kernel(const kernel_id &kernel_id) const noexcept;

    bool has_kernel(const kernel_id &kernel_id, const device &dev) const noexcept;
};

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt) {
    return detail::make_kernel_bundle<State>(
        detail::get_kernel_bundle(ctxt, ctxt.get_devices(), get_kernel_ids(), State));
}

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, const std::vector<kernel_id> &kernel_ids) {
    return detail::make_kernel_bundle<State>(detail::get_kernel_bundle(ctxt, ctxt.get_devices(), kernel_ids, State));
}

template<typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt) {
    return get_kernel_bundle<State>(ctxt, {get_kernel_id<KernelName>()});
}

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, const std::vector<device> &devs) {
    return detail::make_kernel_bundle<State>(detail::get_kernel_bundle(ctxt, devs, get_kernel_ids(), State));
}

template<bundle_state State>
kernel_bundle<State> get_kernel_bundle(
    const context &ctxt, const std::vector<device> &devs, const std::vector<kernel_id> &kernel_ids) {
    return detail::make_kernel_bundle<State>(detail::get_kernel_bundle(ctxt, devs, kernel_ids, State));
}

template<typename KernelName, bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, const std::vector<device> &devs) {
    return get_kernel_bundle<State>(ctxt, devs, {get_kernel_id<KernelName>()});
}

template<bundle_state State, typename Selector>
kernel_bundle<State> get_kernel_bundle(const context &ctxt, Selector selector); // TODO

template<bundle_state State, typename Selector>
kernel_bundle<State> get_kernel_bundle(
    const context &ctxt, const std::vector<device> &devs, Selector selector); //  TODO

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt) {
    return detail::has_kernel_bundle(ctxt, ctxt.get_devices(), get_kernel_ids(), State);
}

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<kernel_id> &kernel_ids) {
    return detail::has_kernel_bundle(ctxt, ctxt.get_devices(), kernel_ids, State);
}

template<typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctxt) {
    return has_kernel_bundle<State>(ctxt, {get_kernel_id<KernelName>()});
}

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs) {
    return detail::has_kernel_bundle(ctxt, devs, get_kernel_ids(), State);
}

template<bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs, const std::vector<kernel_id> &kernel_ids) {
    return detail::has_kernel_bundle(ctxt, devs, kernel_ids, State);
}

template<typename KernelName, bundle_state State>
bool has_kernel_bundle(const context &ctxt, const std::vector<device> &devs) {
    return has_kernel_bundle<State>(ctxt, devs, {get_kernel_id<KernelName>()});
}

inline bool is_compatible(const std::vector<kernel_id> &kernel_ids, const device &dev) {
    (void)kernel_ids;
    (void)dev;
    return true;
}

template<typename KernelName>
bool is_compatible(const device &dev) {
    (void)get_kernel_id<KernelName>(); // throw if non-existent
    (void)dev;
    return true;
}

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

template<simsycl::sycl::bundle_state State>
struct std::hash<simsycl::sycl::device_image<State>>
    : public std::hash<
          simsycl::detail::reference_type<simsycl::sycl::device_image<State>, simsycl::detail::device_image_state>> {};

namespace simsycl::detail {

template<typename KernelName, typename KernelFunc>
inline const sycl::kernel_id registered_kernel_id = register_kernel(typeid(KernelName *), typeid(KernelFunc));

template<typename KernelName, typename KernelFunc>
void register_kernel_on_static_construction() {
    (void)registered_kernel_id<KernelName, KernelFunc>; // instantiate global const
}

} // namespace simsycl::detail
