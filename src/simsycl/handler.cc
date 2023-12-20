#include <simsycl/detail/utils.hh>
#include <simsycl/sycl/device.hh>
#include <simsycl/sycl/exception.hh>
#include <simsycl/sycl/group_functions.hh>
#include <simsycl/sycl/handler.hh>

#include <numeric>

#include <boost/context/continuation.hpp>


namespace simsycl::detail {

void concurrent_nd_item::yield_to_scheduler() { // NOLINT(readability-make-member-function-const)
    *scheduler = scheduler->resume();
}

template<int Dimensions>
void dispatch_for_nd_range(const sycl::device &device, const sycl::nd_range<Dimensions> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<Dimensions> &kernel) //
{
    const auto required_local_memory = std::accumulate(local_memory.begin(), local_memory.end(), size_t{0},
        [](size_t sum, const local_memory_requirement &req) { return sum + req.size; });
    if(required_local_memory > device.get_info<sycl::info::device::local_mem_size>()) {
        throw sycl::exception(sycl::errc::accessor, "total required local memory exceeds device limit");
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
    const auto sub_group_local_linear_range = config::max_sub_group_size;
    const auto sub_group_local_range = sycl::range<1>(sub_group_local_linear_range);
    assert(sub_group_local_linear_range > 0);
    const auto sub_group_linear_range_in_group = detail::div_ceil(local_linear_range, sub_group_local_linear_range);
    const sycl::range<1> sub_group_range_in_group{sub_group_linear_range_in_group};
    assert(sub_group_linear_range_in_group > 0);

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
        const auto sub_group_linear_id_in_group = local_linear_id / sub_group_local_linear_range;
        const auto thread_linear_id_in_sub_group = local_linear_id % sub_group_local_linear_range;
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
                group_linear_range, sub_group_local_range, thread_id_in_sub_group, sub_group_id_in_group,
                sub_group_range_in_group, &concurrent_nd_item, &concurrent_group, &concurrent_sub_group, &kernel,
                &concurrent_items_exited, &caught_exceptions, &range](boost::context::continuation &&scheduler) //
            {
                // yield immediately to allow the scheduling loop to set up local memory pointers
                concurrent_nd_item.scheduler = &scheduler;
                concurrent_nd_item.yield_to_scheduler();

                for(size_t group_linear_id = concurrent_group_idx; group_linear_id < group_linear_range;
                    group_linear_id += num_concurrent_groups) //
                {
                    concurrent_nd_item.instance = nd_item_instance{};
                    // the first item to arrive in this group will create the new group instance
                    if(concurrent_group.instance.group_linear_id != group_linear_id) {
                        concurrent_group.instance = group_instance(group_linear_id);
                        concurrent_sub_group.instance = sub_group_instance{};
                    }

                    const auto group_id = linear_index_to_id(group_range, group_linear_id);
                    const auto global_id = group_id * sycl::id<Dimensions>(local_range) + local_id;

                    SIMSYCL_START_IGNORING_DEPRECATIONS;
                    const auto global_item = detail::make_item(global_id, range.get_global_range(), range.get_offset());
                    SIMSYCL_STOP_IGNORING_DEPRECATIONS
                    const auto local_item = detail::make_item(local_id, range.get_local_range());
                    const auto group_item = detail::make_item(group_id, range.get_group_range());

                    const auto group = detail::make_group(local_item, global_item, group_item, &concurrent_group);
                    const auto sub_group = detail::make_sub_group(thread_id_in_sub_group, sub_group_local_range,
                        sub_group_id_in_group, sub_group_range_in_group, &concurrent_sub_group);
                    const auto nd_item
                        = detail::make_nd_item(global_item, local_item, group, sub_group, &concurrent_nd_item);

                    try {
                        kernel(nd_item);
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
                        concurrent_nd_item.yield_to_scheduler();
                    }
                }

                ++concurrent_items_exited;
                concurrent_nd_item.scheduler = nullptr;
                return std::move(scheduler);
            }));
    }

    // run until all are complete (this does an extra loop)
    while(concurrent_items_exited < num_concurrent_items) {
        for(size_t concurrent_global_idx = 0; concurrent_global_idx < num_concurrent_items; ++concurrent_global_idx) {
            if(!fibers[concurrent_global_idx]) continue; // already exited

            // adjust local memory pointers before switching fibers
            const auto concurrent_group_idx = concurrent_global_idx / local_linear_range;
            for(size_t i = 0; i < local_memory.size(); ++i) {
                *local_memory[i].ptr = concurrent_groups[concurrent_group_idx].local_memory_allocations[i].get();
            }

            fibers[concurrent_global_idx] = fibers[concurrent_global_idx].resume();
        }
    }

    // rethrow any encountered exceptions
    for(auto &exception : caught_exceptions) { std::rethrow_exception(exception); }
}

template void dispatch_for_nd_range<1>(const sycl::device &device, const sycl::nd_range<1> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<1> &kernel);
template void dispatch_for_nd_range<2>(const sycl::device &device, const sycl::nd_range<2> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<2> &kernel);
template void dispatch_for_nd_range<3>(const sycl::device &device, const sycl::nd_range<3> &range,
    const std::vector<local_memory_requirement> &local_memory, const nd_kernel<3> &kernel);

} // namespace simsycl::detail
