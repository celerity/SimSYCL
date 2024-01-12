#include "simsycl/detail/group_operation_impl.hh"

namespace simsycl::detail {

const char *group_operation_id_to_string(group_operation_id id) {
    switch(id) {
        case group_operation_id::broadcast: return "broadcast";
        case group_operation_id::barrier: return "barrier";
        case group_operation_id::joint_any_of: return "joint_any_of";
        case group_operation_id::any_of: return "any_of";
        case group_operation_id::joint_all_of: return "joint_all_of";
        case group_operation_id::all_of: return "all_of";
        case group_operation_id::joint_none_of: return "joint_none_of";
        case group_operation_id::none_of: return "none_of";
        case group_operation_id::shift_left: return "shift_left";
        case group_operation_id::shift_right: return "shift_right";
        case group_operation_id::permute_by_xor: return "permute";
        case group_operation_id::select: return "select";
        case group_operation_id::joint_reduce: return "joint_reduce";
        case group_operation_id::reduce: return "reduce";
        case group_operation_id::joint_exclusive_scan: return "joint_exclusive_scan";
        case group_operation_id::exclusive_scan: return "exclusive_scan";
        case group_operation_id::joint_inclusive_scan: return "joint_inclusive_scan";
        case group_operation_id::inclusive_scan: return "inclusive_scan";
        case group_operation_id::exit: return "exit";
    }
    return "unknown";
}

void check_group_op_validity(
    [[maybe_unused]] int linear_id_in_group, const group_operation_data &new_op, group_operation_data &existing_op) {
    const bool id_equivalent = existing_op.id == new_op.id;
    const bool participant_count_equivalent = existing_op.expected_num_work_items == new_op.expected_num_work_items;
    const bool still_incomplete = existing_op.num_work_items_participating < existing_op.expected_num_work_items;
    existing_op.valid = existing_op.valid && id_equivalent && participant_count_equivalent && still_incomplete;

    SIMSYCL_CHECK_MSG(id_equivalent,
        "group operation id mismatch: group recorded operation \"%s\", but work item #%d is trying to perform \"%s\"",
        group_operation_id_to_string(existing_op.id), linear_id_in_group, group_operation_id_to_string(new_op.id));
    SIMSYCL_CHECK_MSG(participant_count_equivalent,
        "group operation participant count mismatch: group recorded operation \"%s\" with %d participants, but work "
        "item #%d is trying to perform \"%s\" with %d participants",
        group_operation_id_to_string(existing_op.id), existing_op.expected_num_work_items, linear_id_in_group,
        group_operation_id_to_string(new_op.id), new_op.expected_num_work_items);
    SIMSYCL_CHECK_MSG(still_incomplete,
        "group operation already complete: group completed operation \"%s\" with %d participants, but work item #%d is "
        "trying to enter it",
        group_operation_id_to_string(existing_op.id), existing_op.expected_num_work_items, linear_id_in_group);
    SIMSYCL_CHECK_MSG(existing_op.valid, "group operation already invalid");
}

}; // namespace simsycl::detail
