#pragma once

#include <cstddef>
#include <memory>
#include <typeindex>
#include <vector>

#include "simsycl/sycl/enums.hh"

namespace simsycl::detail {

enum class group_operation_id {
    broadcast,
    barrier,
    joint_any_of,
    any_of,
    joint_all_of,
    all_of,
    joint_none_of,
    none_of,
    shift_left,
    shift_right,
    permute,
    select,
    reduce,
    joint_exclusive_scan,
    exclusive_scan,
    joint_inclusive_scan,
    inclusive_scan,
};

// additional data required to implement and check correct use for some group operations

struct group_per_operation_data {
    // non-templated base class for specialized per-operation data
    virtual ~group_per_operation_data() = default;
};

template <typename T>
struct group_broadcast_data : group_per_operation_data {
    size_t local_linear_id = 0;
    std::type_index type = std::type_index(typeid(void));
    std::vector<T> values;
};
struct group_barrier_data : group_per_operation_data {
    sycl::memory_scope fence_scope;
};
struct group_joint_op_data : group_per_operation_data {
    std::intptr_t first;
    std::intptr_t last;
};
struct group_shift_data : group_per_operation_data {
    size_t delta;
};
struct group_permute_data : group_per_operation_data {
    size_t mask;
};
struct group_joint_scan_data : group_per_operation_data {
    std::intptr_t first;
    std::intptr_t last;
    std::intptr_t result;
    std::vector<std::byte> init;
};

struct group_operation_data {
    group_operation_id id;
    size_t expected_num_work_items;
    size_t num_work_items_participating;
    std::unique_ptr<group_per_operation_data> per_op_data;
};

} // namespace simsycl::detail
