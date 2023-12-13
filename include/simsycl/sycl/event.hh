#pragma once

#include "forward.hh"
#include "type_traits.hh"

#include "../detail/reference_type.hh"

#include <chrono>
#include <vector>


namespace simsycl::detail {

struct execution_status {
    std::chrono::steady_clock::time_point t_submit{};
    std::chrono::steady_clock::time_point t_start{};
    std::chrono::steady_clock::time_point t_end{};

    void start() { t_start = std::chrono::steady_clock::now(); }

    [[nodiscard]] static execution_status submit() {
        execution_status status;
        status.t_submit = std::chrono::steady_clock::now();
        return status;
    }

    [[nodiscard]] static execution_status submit_and_start() {
        auto status = submit();
        status.start();
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

class event : detail::reference_type<event, detail::execution_status> {
  public:
    event() = default;

    backend get_backend() const noexcept;

    std::vector<event> get_wait_list();

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
    friend event detail::make_event(const detail::execution_status &status);

    explicit event(const detail::execution_status &status)
        : detail::reference_type<event, detail::execution_status>(std::in_place, status) {}
};

} // namespace simsycl::sycl

namespace simsycl::detail {

inline sycl::event make_event(const execution_status &status) { return sycl::event(status); }

inline sycl::event execution_status::end() {
    t_end = std::chrono::steady_clock::now();
    return make_event(*this);
}

inline sycl::event execution_status::instant() { return submit_and_start().end(); }

} // namespace simsycl::detail
