#include <simsycl/detail/utils.hh>
#include <simsycl/schedule.hh>
#include <simsycl/sycl/device.hh>
#include <simsycl/sycl/exception.hh>
#include <simsycl/sycl/group_functions.hh>
#include <simsycl/sycl/handler.hh>
#include <simsycl/system.hh>

#include <numeric>
#include <random>

#include <boost/context/continuation.hpp>

namespace simsycl {

cooperative_schedule::state round_robin_schedule::init(std::vector<size_t> &order) const {
    std::iota(order.begin(), order.end(), 0);
    return 0;
}

cooperative_schedule::state round_robin_schedule::update(state state_before, std::vector<size_t> &order) const {
    (void)order;
    return state_before;
}

cooperative_schedule::state shuffle_schedule::init(std::vector<size_t> &order) const {
    std::minstd_rand rng(m_seed);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    return rng();
}

cooperative_schedule::state shuffle_schedule::update(state state_before, std::vector<size_t> &order) const {
    std::minstd_rand rng(state_before);
    std::shuffle(order.begin(), order.end(), rng);
    return rng();
}

} // namespace simsycl

namespace simsycl::detail {

template<int Dimensions, typename Offset>
void sequential_for(const sycl::range<Dimensions> &range, const Offset &offset,
    const simple_kernel<Dimensions, with_offset_v<Offset>> &kernel) {
    // limit the number of work items scheduled at a time to avoid allocating huge index buffers
    constexpr size_t max_schedule_chunk_size = 16 << 10;
    const auto schedule_chunk_size = std::min(range.size(), max_schedule_chunk_size);
    const auto &schedule = get_cooperative_schedule();
    std::vector<size_t> order(schedule_chunk_size);
    auto schedule_state = schedule.init(order);

    for(size_t schedule_offset = 0; schedule_offset < range.size(); schedule_offset += max_schedule_chunk_size) {
        for(size_t schedule_id = 0; schedule_id < schedule_chunk_size; ++schedule_id) {
            const auto linear_id = schedule_offset + order[schedule_id];
            if(linear_id < range.size()) {
                if constexpr(with_offset_v<Offset>) {
                    const auto id = offset + linear_index_to_id(range, linear_id);
                    kernel(make_item(id, range, offset));
                } else {
                    const auto id = linear_index_to_id(range, linear_id);
                    kernel(make_item(id, range));
                }
            }
        }
        schedule_state = schedule.update(schedule_state, order);
    }
}

template void sequential_for(
    const sycl::range<1> &range, const no_offset_t & /* no offset */, const simple_kernel<1, false> &kernel);
template void sequential_for(
    const sycl::range<2> &range, const no_offset_t & /* no offset */, const simple_kernel<2, false> &kernel);
template void sequential_for(
    const sycl::range<3> &range, const no_offset_t & /* no offset */, const simple_kernel<3, false> &kernel);
template void sequential_for<1, sycl::id<1>>(
    const sycl::range<1> &range, const sycl::id<1> &offset, const simple_kernel<1, true> &kernel);
template void sequential_for(
    const sycl::range<2> &range, const sycl::id<2> &offset, const simple_kernel<2, true> &kernel);
template void sequential_for(
    const sycl::range<3> &range, const sycl::id<3> &offset, const simple_kernel<3, true> &kernel);

template<int Dimensions>
sycl::range<Dimensions> unit_range() {
    sycl::range<Dimensions> r;
    for(int i = 0; i < Dimensions; ++i) { r[i] = 1; }
    return r;
}

template<int Dimensions>
void sequential_for_work_group(sycl::range<Dimensions> num_work_groups,
    std::optional<sycl::range<Dimensions>> work_group_size, const hierarchical_kernel<Dimensions> &kernel) {
    const auto type
        = work_group_size.has_value() ? group_type::hierarchical_explicit_size : group_type::hierarchical_implicit_size;
    for(size_t group_linear_id = 0; group_linear_id < num_work_groups.size(); ++group_linear_id) {
        const auto group_id = linear_index_to_id(num_work_groups, group_linear_id);
        const auto group_item = make_item(group_id, num_work_groups);
        const auto physical_local_item
            = make_item(sycl::id<Dimensions>(), work_group_size.value_or(unit_range<Dimensions>()));
        const auto global_item = make_item(group_id * sycl::id(physical_local_item.get_range()),
            physical_local_item.get_range() * group_item.get_range(), sycl::id<Dimensions>());
        kernel(make_group(type, physical_local_item, global_item, group_item, nullptr));
    }
}

template void sequential_for_work_group(sycl::range<1> num_work_groups, std::optional<sycl::range<1>> work_group_size,
    const hierarchical_kernel<1> &kernel);
template void sequential_for_work_group(sycl::range<2> num_work_groups, std::optional<sycl::range<2>> work_group_size,
    const hierarchical_kernel<2> &kernel);
template void sequential_for_work_group(sycl::range<3> num_work_groups, std::optional<sycl::range<3>> work_group_size,
    const hierarchical_kernel<3> &kernel);

boost::context::continuation g_scheduler;

void enter_kernel_fiber(boost::context::continuation &&from_scheduler) {
    assert(!g_scheduler && "attempting to enter a nd_range kernel fiber from within another fiber");
    g_scheduler = std::move(from_scheduler);
}

boost::context::continuation &&leave_kernel_fiber() {
    assert(g_scheduler && "attempting to leave a nd_range kernel fiber, but none is active");
    return std::move(g_scheduler);
}

void yield_to_kernel_scheduler() {
    assert(g_scheduler && "attempting to yield from outside a nd_range kernel fiber");
    g_scheduler = g_scheduler.resume();
}

void maybe_yield_to_kernel_scheduler() {
    if(g_scheduler) { g_scheduler = g_scheduler.resume(); }
}

template<int Dimensions>
void cooperative_for_nd_range(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<Dimensions> &kernel) //
{
    if(Dimensions > device.get_info<sycl::info::device::max_work_item_dimensions>()) {
        throw sycl::exception(sycl::errc::nd_range, "Work item dimensionality exceeds device limit");
    }

    const auto required_local_memory = std::accumulate(local_memory.begin(), local_memory.end(), size_t{0},
        [](size_t sum, const local_memory_requirement &req) { return sum + req.size; });
    if(required_local_memory > device.get_info<sycl::info::device::local_mem_size>()) {
        throw sycl::exception(sycl::errc::accessor, "Total required local memory exceeds device limit");
    }

    const auto &global_range = range.get_global_range();
    const auto global_linear_range = global_range.size();
    if(global_linear_range == 0) return;
    const auto &group_range = range.get_group_range();
    const auto group_linear_range = group_range.size();
    assert(group_linear_range > 0);
    const auto &local_range = range.get_local_range();
    const auto local_linear_range = local_range.size();
    assert(local_linear_range > 0);

    if(local_linear_range > device.get_info<sycl::info::device::max_work_group_size>()
        || !all_true(local_range <= device.get_info<sycl::info::device::max_work_item_sizes<Dimensions>>())) {
        throw sycl::exception(sycl::errc::nd_range, "Work group size exceeds device limit");
    }

    const auto sub_group_max_local_linear_range = device.get_info<sycl::info::device::sub_group_sizes>().at(0);
    const auto sub_group_max_local_range = sycl::range<1>(sub_group_max_local_linear_range);
    assert(sub_group_max_local_linear_range > 0);
    const auto sub_group_linear_range_in_group = detail::div_ceil(local_linear_range, sub_group_max_local_linear_range);
    const sycl::range<1> sub_group_range_in_group{sub_group_linear_range_in_group};
    assert(sub_group_linear_range_in_group > 0);

    if(sub_group_linear_range_in_group > device.get_info<sycl::info::device::max_num_sub_groups>()) {
        throw sycl::exception(sycl::errc::nd_range, "Number of sub-groups in work group exceeds device limit");
    }

    // limit the number of concurrent groups to avoid allocating excessive numbers of fibers
    const size_t max_num_concurrent_groups = device.get_info<sycl::info::device::max_compute_units>();
    const auto num_concurrent_groups = std::min(group_linear_range, max_num_concurrent_groups);
    const auto num_concurrent_sub_groups = num_concurrent_groups * sub_group_linear_range_in_group;
    const auto num_concurrent_items = num_concurrent_groups * local_linear_range;

    std::vector<detail::concurrent_group> concurrent_groups(num_concurrent_groups);
    std::vector<detail::concurrent_sub_group> concurrent_sub_groups(num_concurrent_sub_groups);
    std::vector<detail::concurrent_nd_item> num_concurrent_nd_items(num_concurrent_items);

    for(auto &cgroup : concurrent_groups) {
        cgroup.local_memory_allocations.resize(local_memory.size());
        for(size_t i = 0; i < local_memory.size(); ++i) {
            cgroup.local_memory_allocations[i] = allocation(local_memory[i].size, local_memory[i].align);
        }
    }

    size_t concurrent_items_exited = 0;
    std::vector<std::exception_ptr> caught_exceptions;
    std::vector<boost::context::continuation> fibers;

    // build the item / group structures and fibers for all concurrent work items
    for(size_t concurrent_global_idx = 0; concurrent_global_idx < num_concurrent_items; ++concurrent_global_idx) {
        // all these ids and linear ids are persistent between all groups this fiber iterates over
        const auto local_linear_id = concurrent_global_idx % local_linear_range;
        const auto local_id = linear_index_to_id(local_range, local_linear_id);
        const auto sub_group_linear_id_in_group = local_linear_id / sub_group_max_local_linear_range;
        const auto thread_linear_id_in_sub_group = local_linear_id % sub_group_max_local_linear_range;
        const auto sub_group_id_in_group = sycl::id<1>(sub_group_linear_id_in_group);
        const auto thread_id_in_sub_group = sycl::id<1>(thread_linear_id_in_sub_group);

        const auto concurrent_group_idx = concurrent_global_idx / local_linear_range;
        const auto concurrent_sub_group_idx
            = concurrent_group_idx * sub_group_linear_range_in_group + sub_group_linear_id_in_group;

        auto &concurrent_nd_item = num_concurrent_nd_items[concurrent_global_idx];

        auto &concurrent_group = concurrent_groups[concurrent_group_idx];
        concurrent_group.concurrent_nd_items.push_back(&concurrent_nd_item);
        concurrent_nd_item.concurrent_group = &concurrent_group;

        auto &concurrent_sub_group = concurrent_sub_groups[concurrent_sub_group_idx];
        concurrent_sub_group.concurrent_nd_items.push_back(&concurrent_nd_item);

        fibers.push_back(boost::context::callcc(
            [concurrent_group_idx, num_concurrent_groups, local_id, local_range, local_linear_range, group_range,
                group_linear_range, sub_group_linear_id_in_group, sub_group_linear_range_in_group,
                sub_group_max_local_linear_range, sub_group_max_local_range, thread_id_in_sub_group,
                sub_group_id_in_group, sub_group_range_in_group, &concurrent_nd_item, &concurrent_group,
                &concurrent_sub_group, &kernel, &concurrent_items_exited, &caught_exceptions,
                &range](boost::context::continuation &&scheduler) //
            {
                // yield immediately to allow the scheduling loop to set up local memory pointers
                enter_kernel_fiber(std::move(scheduler));
                yield_to_kernel_scheduler();

                for(size_t group_linear_id = concurrent_group_idx; group_linear_id < group_linear_range;
                    group_linear_id += num_concurrent_groups) //
                {
                    const auto sub_group_linear_id
                        = group_linear_id * sub_group_linear_range_in_group + sub_group_linear_id_in_group;

                    concurrent_nd_item.instance = nd_item_instance{};
                    // the first item to arrive in this group will create the new group instance
                    if(concurrent_group.instance.group_linear_id != group_linear_id) {
                        concurrent_group.instance = group_instance(group_linear_id);
                    }
                    // the first item to arrive in this sub_group will create the new sub_group instance
                    if(concurrent_sub_group.instance.sub_group_linear_id != sub_group_linear_id) {
                        concurrent_sub_group.instance = sub_group_instance(sub_group_linear_id);
                    }

                    const auto group_id = linear_index_to_id(group_range, group_linear_id);
                    const auto global_id = group_id * sycl::id<Dimensions>(local_range) + local_id;

                    // if sub-group range is not divisible by local range, the last sub-group will be smaller
                    const auto sub_group_local_linear_range = std::min(sub_group_max_local_linear_range,
                        local_linear_range - sub_group_linear_id_in_group * sub_group_max_local_linear_range);
                    const auto sub_group_local_range = sycl::range<1>(sub_group_local_linear_range);

                    SIMSYCL_START_IGNORING_DEPRECATIONS;
                    const auto global_item = detail::make_item(global_id, range.get_global_range(), range.get_offset());
                    SIMSYCL_STOP_IGNORING_DEPRECATIONS
                    const auto local_item = detail::make_item(local_id, range.get_local_range());
                    const auto group_item = detail::make_item(group_id, range.get_group_range());

                    const auto group = detail::make_group(
                        group_type::nd_range, local_item, global_item, group_item, &concurrent_group);
                    const auto sub_group = detail::make_sub_group(thread_id_in_sub_group, sub_group_local_range,
                        sub_group_max_local_range, sub_group_id_in_group, sub_group_range_in_group,
                        &concurrent_sub_group);
                    const auto nd_item
                        = detail::make_nd_item(global_item, local_item, group, sub_group, &concurrent_nd_item);

                    try {
                        kernel(nd_item);
                        // Add an implicit "exit" operations to groups and sub-groups to catch potential divergence on
                        // the last group operation
                        perform_group_operation(
                            nd_item.get_group(), detail::group_operation_id::exit, detail::group_operation_spec{});
                        perform_group_operation(
                            nd_item.get_sub_group(), detail::group_operation_id::exit, detail::group_operation_spec{});
                    } catch(...) { //
                        caught_exceptions.push_back(std::current_exception());
                    }

                    // Wait for all items in the group before scheduling the next group on this fiber (otherwise we
                    // could get races between items of different groups accessing the same re-used local memory
                    // allocation).
                    ++concurrent_group.instance.num_items_exited;
                    // If group_linear_id changes, another fiber has advanced to the next group, if we observe that all
                    // items have exited, we are the fiber to proceed to the next iteration.
                    while(concurrent_group.instance.group_linear_id == group_linear_id
                        && concurrent_group.instance.num_items_exited < local_linear_range) {
                        yield_to_kernel_scheduler();
                    }
                }

                ++concurrent_items_exited;
                return leave_kernel_fiber();
            }));
    }

    const auto &schedule = get_cooperative_schedule();
    std::vector<size_t> order(num_concurrent_items);
    auto schedule_state = schedule.init(order);

    // run until all are complete (this does an extra loop)
    while(concurrent_items_exited < num_concurrent_items) {
        for(size_t i = 0; i < num_concurrent_items; ++i) {
            const size_t concurrent_global_idx = order[i];

            if(!fibers[concurrent_global_idx]) continue; // already exited

            // adjust local memory pointers before switching fibers
            const auto concurrent_group_idx = concurrent_global_idx / local_linear_range;
            for(size_t i = 0; i < local_memory.size(); ++i) {
                *local_memory[i].ptr = concurrent_groups[concurrent_group_idx].local_memory_allocations[i].get();
            }

            fibers[concurrent_global_idx] = fibers[concurrent_global_idx].resume();
        }
        schedule_state = schedule.update(schedule_state, order);
    }

    // rethrow any encountered exceptions
    for(auto &exception : caught_exceptions) { std::rethrow_exception(exception); }
}

template void cooperative_for_nd_range<1>(const sycl::device &device, const sycl::nd_range<1> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<1> &kernel);
template void cooperative_for_nd_range<2>(const sycl::device &device, const sycl::nd_range<2> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<2> &kernel);
template void cooperative_for_nd_range<3>(const sycl::device &device, const sycl::nd_range<3> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<3> &kernel);

template<int Dimensions>
std::vector<allocation> prepare_hierarchical_parallel_for(const sycl::device &device,
    std::optional<sycl::range<Dimensions>> work_group_size,
    const std::vector<local_memory_requirement> &local_memory) //
{
    if(Dimensions > device.get_info<sycl::info::device::max_work_item_dimensions>()) {
        throw sycl::exception(sycl::errc::invalid, "Work item dimensionality exceeds device limit");
    }

    const auto required_local_memory = std::accumulate(local_memory.begin(), local_memory.end(), size_t{0},
        [](size_t sum, const local_memory_requirement &req) { return sum + req.size; });
    if(required_local_memory > device.get_info<sycl::info::device::local_mem_size>()) {
        throw sycl::exception(sycl::errc::accessor, "Total required local memory exceeds device limit");
    }

    if(work_group_size.has_value()) {
        const auto &local_range = *work_group_size;
        const auto local_linear_range = local_range.size();
        assert(local_linear_range > 0);

        if(local_linear_range > device.get_info<sycl::info::device::max_work_group_size>()
            || !all_true(local_range <= device.get_info<sycl::info::device::max_work_item_sizes<Dimensions>>())) {
            throw sycl::exception(sycl::errc::invalid, "Work group size exceeds device limit");
        }
    }

    std::vector<allocation> local_allocations;
    for(size_t i = 0; i < local_memory.size(); ++i) {
        *local_memory[i].ptr = local_allocations.emplace_back(local_memory[i].size, local_memory[i].align).get();
    }
    return local_allocations;
}

template std::vector<allocation> prepare_hierarchical_parallel_for<1>(const sycl::device &device,
    std::optional<sycl::range<1>> work_group_size, const std::vector<local_memory_requirement> &local_memory);
template std::vector<allocation> prepare_hierarchical_parallel_for<2>(const sycl::device &device,
    std::optional<sycl::range<2>> work_group_size, const std::vector<local_memory_requirement> &local_memory);
template std::vector<allocation> prepare_hierarchical_parallel_for<3>(const sycl::device &device,
    std::optional<sycl::range<3>> work_group_size, const std::vector<local_memory_requirement> &local_memory);

} // namespace simsycl::detail
