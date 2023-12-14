#pragma once

#include "check.hh"

#include <memory>

namespace simsycl::detail {

template<typename Derived, typename State>
class reference_type;

}

template<typename Derived, typename State>
struct std::hash<simsycl::detail::reference_type<Derived, State>> {
    size_t operator()(const Derived &rt) { return static_cast<size_t>(reinterpret_cast<uintptr_t>(rt.m_state.get())); }
};

namespace simsycl::detail {

template<typename Derived, typename State>
class weak_ref {
  public:
    weak_ref() = default;

    weak_ref(std::weak_ptr<State> &&state) : m_state(std::move(state)) {}

    Derived lock() const { return Derived(m_state.lock()); }

  private:
    std::weak_ptr<State> m_state;
};

template<typename Derived, typename State>
class reference_type {
  public:
    friend bool operator==(const Derived &lhs, const Derived &rhs) { return lhs.m_state.get() == rhs.m_state.get(); }
    friend bool operator!=(const Derived &lhs, const Derived &rhs) { return lhs.m_state.get() != rhs.m_state.get(); }

  protected:
    using state_type = State;

    reference_type() = default;

    reference_type(std::shared_ptr<state_type> &&state) : m_state(std::move(state)) {
        SIMSYCL_CHECK(m_state != nullptr);
    }

    template<typename... CtorParams>
    explicit reference_type(std::in_place_t /* tag */, CtorParams &&...ctor_args)
        : m_state(std::make_shared<State>(std::forward<CtorParams>(ctor_args)...)) {
        static_assert(std::is_base_of_v<reference_type, Derived>);
    }

    state_type &state() {
        SIMSYCL_CHECK(m_state != nullptr);
        return *m_state;
    }

    const state_type &state() const {
        SIMSYCL_CHECK(m_state != nullptr);
        return *m_state;
    }

    detail::weak_ref<Derived, State> weak_ref() {
        SIMSYCL_CHECK(m_state != nullptr);
        return detail::weak_ref<Derived, State>(std::weak_ptr<state_type>(m_state));
    }

  private:
    friend struct std::hash<reference_type<Derived, State>>;

    template<typename, typename>
    friend class weak_ref;

    std::shared_ptr<state_type> m_state;
};

} // namespace simsycl::detail
