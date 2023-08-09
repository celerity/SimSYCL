#pragma once

#include <memory>

namespace simsycl::detail {

template <typename Derived, typename State>
class reference_type;

}

template <typename Derived, typename State>
struct std::hash<simsycl::detail::reference_type<Derived, State>> {
    size_t operator()(const Derived &rt) { return static_cast<size_t>(reinterpret_cast<uintptr_t>(rt.m_state.get())); }
};

namespace simsycl::detail {

template <typename Derived, typename State>
class reference_type {
  public:
    friend bool operator==(const Derived &lhs, const Derived &rhs) { return lhs.m_state.get() == rhs.m_state.get(); }
    friend bool operator!=(const Derived &lhs, const Derived &rhs) { return lhs.m_state.get() != rhs.m_state.get(); }

  protected:
    using state_type = State;

    explicit reference_type(state_type &&st) : m_state(std::make_shared(std::move(st))) {
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

  private:
    friend struct std::hash<reference_type<Derived, State>>;
    std::shared_ptr<state_type> m_state;
};

} // namespace simsycl::detail
