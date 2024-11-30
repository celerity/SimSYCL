#pragma once

#include "forward.hh"
#include "info.hh"
#include "type_traits.hh"

#include "../detail/reference_type.hh"

#include <chrono>
#include <vector>


namespace simsycl::detail {

struct event_state {
    std::chrono::steady_clock::time_point t_submit;
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_end;

    void start() { t_start = std::chrono::steady_clock::now(); }

    [[nodiscard]] static event_state submit() {
        event_state status;
        status.t_submit = std::chrono::steady_clock::now();
        return status;
    }

    [[nodiscard]] sycl::event end();

    [[nodiscard]] static sycl::event instant();
};

template<typename Clock, typename Dur>
uint64_t nanoseconds_since_epoch(std::chrono::time_point<Clock, Dur> time_point) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time_point.time_since_epoch()).count();
}

} // namespace simsycl::detail

namespace simsycl::sycl {

class event : public detail::reference_type<event, detail::event_state> {
  public:
    event() = default;

    backend get_backend() const noexcept { return backend::simsycl; }

    std::vector<event> get_wait_list() {
        // spec: already completed events do not need to be included
        return {};
    }

    void wait() {}

    static void wait(const std::vector<event> & /* event_list */) {}

    void wait_and_throw() {}

    static void wait_and_throw(const std::vector<event> & /* event_list */) {}

    template<typename Param>
    typename Param::return_type get_info() const {
        if constexpr(std::is_same_v<Param, info::event::command_execution_status>) {
            return info::event_command_status::complete;
        } else {
            static_assert(detail::always_false<Param>, "Unknown event::get_info() parameter");
        }
    }

    template<typename Param>
    typename Param::return_type get_backend_info() const {
        static_assert(detail::always_false<Param>, "Unknown event::get_backend_info() parameter");
    }

    template<typename Param>
    typename Param::return_type get_profiling_info() const {
        if constexpr(std::is_same_v<Param, info::event_profiling::command_submit>) {
            return detail::nanoseconds_since_epoch(state().t_submit);
        } else if constexpr(std::is_same_v<Param, info::event_profiling::command_start>) {
            return detail::nanoseconds_since_epoch(state().t_start);
        } else if constexpr(std::is_same_v<Param, info::event_profiling::command_end>) {
            return detail::nanoseconds_since_epoch(state().t_end);
        } else {
            static_assert(detail::always_false<Param>, "Unknown event::get_profiling_info() parameter");
        }
    }

  private:
    template<typename>
    friend class detail::weak_ref;

    friend event detail::make_event(std::shared_ptr<detail::event_state> &&state);

    explicit event(std::shared_ptr<detail::event_state> &&state)
        : detail::reference_type<event, detail::event_state>(std::move(state)) {}
};

} // namespace simsycl::sycl


template<>
struct std::hash<simsycl::sycl::event>
    : std::hash<simsycl::detail::reference_type<simsycl::sycl::event, simsycl::detail::event_state>> {};

namespace simsycl::detail {

inline sycl::event make_event(std::shared_ptr<event_state> &&state) { return sycl::event(std::move(state)); }
inline sycl::event make_event(const event_state &state) { return make_event(std::make_shared<event_state>(state)); }

inline sycl::event event_state::end() {
    t_end = std::chrono::steady_clock::now();
    return make_event(*this);
}

inline sycl::event event_state::instant() {
    const auto now = std::chrono::steady_clock::now();
    return make_event(event_state{now, now, now});
}

} // namespace simsycl::detail
