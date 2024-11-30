#pragma once

#include <cassert>
#include <mutex>
#include <utility>


namespace simsycl::detail {

class system_lock {
  public:
    system_lock();

  private:
    template<typename F>
    friend decltype(auto) with_system_lock(F &&f);

    std::lock_guard<std::recursive_mutex> m_lock;
};

template<typename T>
class shared_value {
  public:
    shared_value() : m_value() {}
    shared_value(const T &v) : m_value(v) {}
    shared_value(T &&v) : m_value(std::move(v)) {}

    template<typename... Params>
    shared_value(const std::in_place_t /* tag */, Params &&...params) : m_value(std::forward<Params>(params)...) {}

    shared_value(const shared_value &) = delete;
    shared_value(shared_value &&) = delete;
    shared_value &operator=(const shared_value &) = delete;
    shared_value &operator=(shared_value &&) = delete;

    ~shared_value() = default;

    T &with(system_lock & /* lock */) { return m_value; }

  private:
    T m_value;
};

} // namespace simsycl::detail
