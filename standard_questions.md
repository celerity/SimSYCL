
 * Should group operations allow calling the same operation with a different signature if the semantics are identical?
 * add [[nodiscard]] to spec functions where it makes sense? (ie. group ops)
 * Should group ops like joint any really be restricted to pointers rather than things which can be iterated over?
 * For group scans: maybe change standard wording for init to make it more clear that this is at the "front/left".
 * Should it be allowed to call leader() for groups in a parallel_for_work_item context?
