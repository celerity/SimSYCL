#pragma once

#include "enums.hh"
#include "forward.hh"
#include "property.hh"

#include "simsycl/detail/allocation.hh"

namespace simsycl::sycl {

template <typename T, usm::alloc AllocKind, size_t Alignment = 0>
class usm_allocator {
  public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    template <typename U>
    struct rebind {
        typedef usm_allocator<U, AllocKind, Alignment> other;
    };

    usm_allocator() = delete;

    usm_allocator(const context &sycl_context, const device &sycl_device, const property_list &prop_list = {}) {
        (void)sycl_context;
        (void)sycl_device;
        (void)prop_list;
    }

    usm_allocator(const queue &sycl_queue, const property_list &prop_list = {}) {
        (void)sycl_queue;
        (void)prop_list;
    }

    template <class U>
    usm_allocator(const usm_allocator<U, AllocKind, Alignment> &other) noexcept {
        (void)other;
    }

    usm_allocator(const usm_allocator &other) = default;
    usm_allocator(usm_allocator &&) noexcept;
    usm_allocator &operator=(const usm_allocator &) = default;
    usm_allocator &operator=(usm_allocator &&) = default;

    /// Allocate memory
    T *allocate(size_t count) { return static_cast<T *>(detail::aligned_alloc(Alignment, count * sizeof(T))); }

    /// Deallocate memory
    void deallocate(T *ptr, size_t count) {
        (void)count;
        detail::aligned_free(ptr);
    }

    /// Equality Comparison
    ///
    /// Allocators only compare equal if they are of the same USM kind, alignment, context, and device
    template <class U, usm::alloc AllocKindU, size_t AlignmentU>
    friend bool operator==(
        const usm_allocator<T, AllocKind, Alignment> &lhs, const usm_allocator<U, AllocKindU, AlignmentU> &rhs);

    /// Inequality Comparison
    /// Allocators only compare unequal if they are not of the same USM kind, alignment, context, or device
    template <class U, usm::alloc AllocKindU, size_t AlignmentU>
    friend bool operator!=(
        const usm_allocator<T, AllocKind, Alignment> &lhs, const usm_allocator<U, AllocKindU, AlignmentU> &rhs);
};


inline void *malloc_device(
    size_t num_bytes, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc_device(
    size_t count, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *malloc_device(size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc_device(size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *aligned_alloc_device(size_t alignment, size_t num_bytes, const device &sycl_device,
    const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc_device(size_t alignment, size_t count, const device &sycl_device, const context &sycl_context,
    const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
}

inline void *aligned_alloc_device(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc_device(size_t alignment, size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
};

inline void *malloc_host(size_t num_bytes, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_context;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc_host(size_t count, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_context;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *malloc_host(size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc_host(size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *aligned_alloc_host(
    size_t alignment, size_t num_bytes, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_context;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc_host(
    size_t alignment, size_t count, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_context;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
}

inline void *aligned_alloc_host(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
void *aligned_alloc_host(size_t alignment, size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
}

inline void *malloc_shared(
    size_t num_bytes, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc_shared(
    size_t count, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *malloc_shared(size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc_shared(size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *aligned_alloc_shared(size_t alignment, size_t num_bytes, const device &sycl_device,
    const context &sycl_context, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc_shared(size_t alignment, size_t count, const device &sycl_device, const context &sycl_context,
    const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
}

inline void *aligned_alloc_shared(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc_shared(size_t alignment, size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
}

inline void *malloc(size_t num_bytes, const device &sycl_device, const context &sycl_context, usm::alloc kind,
    const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)kind;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc(size_t count, const device &sycl_device, const context &sycl_context, usm::alloc kind,
    const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)kind;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *malloc(size_t num_bytes, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)kind;
    (void)prop_list;
    return detail::aligned_alloc(1, num_bytes);
}

template <typename T>
T *malloc(size_t count, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)kind;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignof(T), count * sizeof(T)));
}

inline void *aligned_alloc(size_t alignment, size_t num_bytes, const device &sycl_device, const context &sycl_context,
    usm::alloc kind, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)kind;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc(size_t alignment, size_t count, const device &sycl_device, const context &sycl_context,
    usm::alloc kind, const property_list &prop_list = {}) {
    (void)sycl_device;
    (void)sycl_context;
    (void)kind;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
}

inline void *aligned_alloc(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)kind;
    (void)prop_list;
    return detail::aligned_alloc(alignment, num_bytes);
}

template <typename T>
T *aligned_alloc(
    size_t alignment, size_t count, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)sycl_queue;
    (void)kind;
    (void)prop_list;
    return static_cast<T *>(detail::aligned_alloc(alignment, count * sizeof(T)));
}

inline void free(void *ptr, const context &sycl_context) {
    (void)sycl_context;
    detail::aligned_free(ptr);
}

inline void free(void *ptr, const queue &sycl_queue) {
    (void)sycl_queue;
    detail::aligned_free(ptr);
}

usm::alloc get_pointer_type(const void *ptr, const context &sycl_context);

device get_pointer_device(const void *ptr, const context &sycl_context);

} // namespace simsycl::sycl
