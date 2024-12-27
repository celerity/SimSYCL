#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace simsycl {

/// A schedule generates execution orders for work items within the constraints of group synchronization.
///
/// The kernel function is invoked once for each work item as prescribed by the schedule. For ND-range kernels, they can
/// be suspended and resumed multiple times around group collective functions, in which case the schedule can instruct a
/// different order for each iteration.
class cooperative_schedule {
  public:
    using state = uint64_t;

    cooperative_schedule() = default;
    cooperative_schedule(const cooperative_schedule &) = delete;
    cooperative_schedule(cooperative_schedule &&) = delete;
    cooperative_schedule &operator=(const cooperative_schedule &) = delete;
    cooperative_schedule &operator=(cooperative_schedule &&) = delete;
    virtual ~cooperative_schedule() = default;

    /// Fill a vector of linear work item indices for the initial round of kernel invocations.
    ///
    /// The retured `state` is to be carried into the first invocation of `update()`, if any.
    [[nodiscard]] virtual state init(std::vector<size_t> &order) const = 0;

    /// After work items have been suspended on an ND-range kernel collective function, fill the index vector with the
    /// next sequence of work item indices.
    ///
    /// The retured `state` is to be carried into the next invocation of `update()`.
    [[nodiscard]] virtual state update(state state_before, std::vector<size_t> &order) const = 0;
};

/// A schedule executing threads in-order by linear thread id.
class round_robin_schedule final : public cooperative_schedule {
  public:
    [[nodiscard]] state init(std::vector<size_t> &order) const override;
    [[nodiscard]] state update(state state_before, std::vector<size_t> &order) const override;
};

/// A schedule executing threads in randomly shuffled order. After each collective barrier, indices are re-shuffled.
///
/// SYCL programs need to ensure their kernels are correct under any order of thread items, so using a
/// `shuffle_schedule` (potentially with multiple seeds) allows fuzzing that assumption.
class shuffle_schedule final : public cooperative_schedule {
  public:
    shuffle_schedule() = default;
    explicit shuffle_schedule(uint64_t seed) : m_seed(seed) {}

    [[nodiscard]] state init(std::vector<size_t> &order) const override;
    [[nodiscard]] state update(state state_before, std::vector<size_t> &order) const override;

  private:
    uint64_t m_seed = 1234567890;
};

/// Return the thread-locally active schedule.
const cooperative_schedule &get_cooperative_schedule();

/// Set the thread-locally active schedule for future kernel invocations.
///
/// Must not be called from within a kernel.
void set_cooperative_schedule(std::shared_ptr<const cooperative_schedule> schedule);

} // namespace simsycl
