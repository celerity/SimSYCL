#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace simsycl {

class cooperative_schedule {
  public:
    using state = unsigned int;

    cooperative_schedule() = default;
    cooperative_schedule(const cooperative_schedule &) = delete;
    cooperative_schedule(cooperative_schedule &&) = delete;
    cooperative_schedule &operator=(const cooperative_schedule &) = delete;
    cooperative_schedule &operator=(cooperative_schedule &&) = delete;
    virtual ~cooperative_schedule() = default;

    [[nodiscard]] virtual state init(std::vector<size_t> &order) const = 0;
    [[nodiscard]] virtual state update(state state_before, std::vector<size_t> &order) const = 0;
};

class round_robin_schedule final : public cooperative_schedule {
  public:
    [[nodiscard]] state init(std::vector<size_t> &order) const override;
    [[nodiscard]] state update(state state_before, std::vector<size_t> &order) const override;
};

class shuffle_schedule final : public cooperative_schedule {
  public:
    shuffle_schedule() = default;
    explicit shuffle_schedule(unsigned int seed) : m_seed(seed) {}

    [[nodiscard]] state init(std::vector<size_t> &order) const override;
    [[nodiscard]] state update(state state_before, std::vector<size_t> &order) const override;

  private:
    unsigned int m_seed = 1234567890;
};

const cooperative_schedule &get_cooperative_schedule();
void set_cooperative_schedule(std::unique_ptr<cooperative_schedule> schedule);

} // namespace simsycl
