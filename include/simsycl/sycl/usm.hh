#pragma once

#include "device.hh"
#include "enums.hh"
#include "forward.hh"
#include "property.hh"
#include "queue.hh"

#include "simsycl/detail/allocation.hh"

#include <numeric>


namespace simsycl::detail {
template<std::convertible_to<size_t>... Alignments>
constexpr size_t least_common_alignment(const Alignments &...aligns) {
    size_t common = 1;
    ((common = std::lcm(common, std::max<size_t>(1, aligns))), ...);
    return common;
}

} // namespace simsycl::detail

namespace simsycl::sycl {

template<typename T, usm::alloc AllocKind, size_t Alignment = 0>
class usm_allocator {
    static_assert(AllocKind == usm::alloc::host || AllocKind == usm::alloc::shared,
        "Allocations made by sycl::usm_allocator must be host-accessible");

  public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    template<typename U>
    struct rebind {
        typedef usm_allocator<U, AllocKind, Alignment> other;
    };

    usm_allocator() = delete;

    usm_allocator(const context &sycl_context, const device &sycl_device, const property_list &prop_list = {})
        : m_context(sycl_context), m_device(sycl_device) {
        (void)prop_list;
    }

    usm_allocator(const queue &sycl_queue, const property_list &prop_list = {})
        : m_context(sycl_queue.get_context()), m_device(sycl_queue.get_device()) {
        (void)prop_list;
    }

    template<class U>
    usm_allocator(const usm_allocator<U, AllocKind, Alignment> &other) noexcept
        : m_context(other.m_context), m_device(other.m_device) {}

    usm_allocator(const usm_allocator &other) = default;
    usm_allocator(usm_allocator &&other) = default;
    usm_allocator &operator=(const usm_allocator &) = default;
    usm_allocator &operator=(usm_allocator &&) = default;

    T *allocate(size_t count) {
        const auto ptr = static_cast<T *>(detail::usm_alloc(
            m_context, AllocKind, m_device, count * sizeof(T), detail::least_common_alignment(alignof(T), Alignment)));
        if(ptr == nullptr) throw std::bad_alloc();
        return ptr;
    }

    void deallocate(T *ptr, size_t count) {
        (void)count;
        detail::usm_free(ptr, m_context);
    }

    template<class U, usm::alloc AllocKindU, size_t AlignmentU>
    friend bool operator==(
        const usm_allocator<T, AllocKind, Alignment> &lhs, const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
        return AllocKindU == AllocKind && AlignmentU == Alignment && lhs.m_context == rhs.m_context
            && lhs.m_device == rhs.m_device;
    }

    template<class U, usm::alloc AllocKindU, size_t AlignmentU>
    friend bool operator!=(
        const usm_allocator<T, AllocKind, Alignment> &lhs, const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
        return !(lhs == rhs);
    }

  private:
    template<typename, usm::alloc, size_t>
    friend class usm_allocator;

    context m_context;
    device m_device;
};


inline void *malloc_device(
    size_t num_bytes, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, usm::alloc::device, sycl_device, num_bytes, 1);
}

template<typename T>
T *malloc_device(
    size_t count, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(
        detail::usm_alloc(sycl_context, usm::alloc::device, sycl_device, count * sizeof(T), alignof(T)));
}

inline void *malloc_device(size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_queue.get_context(), usm::alloc::device, sycl_queue.get_device(), num_bytes, 1);
}

template<typename T>
T *malloc_device(size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(
        sycl_queue.get_context(), usm::alloc::device, sycl_queue.get_device(), count * sizeof(T), alignof(T)));
}

inline void *aligned_alloc_device(size_t alignment, size_t num_bytes, const device &sycl_device,
    const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, usm::alloc::device, sycl_device, num_bytes, alignment);
}

template<typename T>
T *aligned_alloc_device(size_t alignment, size_t count, const device &sycl_device, const context &sycl_context,
    const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_context, usm::alloc::device, sycl_device, count * sizeof(T),
        detail::least_common_alignment(alignment, alignof(T))));
}

inline void *aligned_alloc_device(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(
        sycl_queue.get_context(), usm::alloc::device, sycl_queue.get_device(), num_bytes, alignment);
}

template<typename T>
T *aligned_alloc_device(size_t alignment, size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_queue.get_context(), usm::alloc::device, sycl_queue.get_device(),
        count * sizeof(T), detail::least_common_alignment(alignment, alignof(T))));
};

inline void *malloc_host(size_t num_bytes, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, usm::alloc::host, std::nullopt, num_bytes, 1);
}

template<typename T>
T *malloc_host(size_t count, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(
        detail::usm_alloc(sycl_context, usm::alloc::host, std::nullopt, count * sizeof(T), alignof(T)));
}

inline void *malloc_host(size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_queue.get_context(), usm::alloc::host, std::nullopt, num_bytes, 1);
}

template<typename T>
T *malloc_host(size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(
        detail::usm_alloc(sycl_queue.get_context(), usm::alloc::host, std::nullopt, count * sizeof(T), alignof(T)));
}

inline void *aligned_alloc_host(
    size_t alignment, size_t num_bytes, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, usm::alloc::host, std::nullopt, num_bytes, alignment);
}

template<typename T>
T *aligned_alloc_host(
    size_t alignment, size_t count, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_context, usm::alloc::host, std::nullopt, count * sizeof(T),
        detail::least_common_alignment(alignment, alignof(T))));
}

inline void *aligned_alloc_host(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_queue.get_context(), usm::alloc::host, std::nullopt, num_bytes, alignment);
}

template<typename T>
T *aligned_alloc_host(size_t alignment, size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_queue.get_context(), usm::alloc::host, std::nullopt,
        count * sizeof(T), detail::least_common_alignment(alignment, alignof(T))));
}

inline void *malloc_shared(
    size_t num_bytes, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, usm::alloc::shared, sycl_device, num_bytes, 1);
}

template<typename T>
T *malloc_shared(
    size_t count, const device &sycl_device, const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(
        detail::usm_alloc(sycl_context, usm::alloc::shared, sycl_device, count * sizeof(T), alignof(T)));
}

inline void *malloc_shared(size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_queue.get_context(), usm::alloc::shared, sycl_queue.get_device(), num_bytes, 1);
}

template<typename T>
T *malloc_shared(size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(
        sycl_queue.get_context(), usm::alloc::shared, sycl_queue.get_device(), count * sizeof(T), alignof(T)));
}

inline void *aligned_alloc_shared(size_t alignment, size_t num_bytes, const device &sycl_device,
    const context &sycl_context, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, usm::alloc::shared, sycl_device, num_bytes, alignment);
}

template<typename T>
T *aligned_alloc_shared(size_t alignment, size_t count, const device &sycl_device, const context &sycl_context,
    const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_context, usm::alloc::shared, sycl_device, count * sizeof(T),
        detail::least_common_alignment(alignment, alignof(T))));
}

inline void *aligned_alloc_shared(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(
        sycl_queue.get_context(), usm::alloc::shared, sycl_queue.get_device(), num_bytes, alignment);
}

template<typename T>
T *aligned_alloc_shared(size_t alignment, size_t count, const queue &sycl_queue, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_queue.get_context(), usm::alloc::shared, sycl_queue.get_device(),
        count * sizeof(T), detail::least_common_alignment(alignment, alignof(T))));
};

inline void *malloc(size_t num_bytes, const device &sycl_device, const context &sycl_context, usm::alloc kind,
    const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, kind, sycl_device, num_bytes, 1);
}

template<typename T>
T *malloc(size_t count, const device &sycl_device, const context &sycl_context, usm::alloc kind,
    const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_context, kind, sycl_device, count * sizeof(T), alignof(T)));
}

inline void *malloc(size_t num_bytes, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_queue.get_context(), kind,
        kind != usm::alloc::host ? std::optional(sycl_queue.get_device()) : std::nullopt, num_bytes, 1);
}

template<typename T>
T *malloc(size_t count, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_queue.get_context(), kind,
        kind != usm::alloc::host ? std::optional(sycl_queue.get_device()) : std::nullopt, count * sizeof(T),
        alignof(T)));
}

inline void *aligned_alloc(size_t alignment, size_t num_bytes, const device &sycl_device, const context &sycl_context,
    usm::alloc kind, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_context, kind, sycl_device, num_bytes, alignment);
}

template<typename T>
T *aligned_alloc(size_t alignment, size_t count, const device &sycl_device, const context &sycl_context,
    usm::alloc kind, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(
        sycl_context, kind, sycl_device, count * sizeof(T), detail::least_common_alignment(alignment, alignof(T))));
}

inline void *aligned_alloc(
    size_t alignment, size_t num_bytes, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)prop_list;
    return detail::usm_alloc(sycl_queue.get_context(), kind,
        kind != usm::alloc::host ? std::optional(sycl_queue.get_device()) : std::nullopt, num_bytes, alignment);
}

template<typename T>
T *aligned_alloc(
    size_t alignment, size_t count, const queue &sycl_queue, usm::alloc kind, const property_list &prop_list = {}) {
    (void)prop_list;
    return static_cast<T *>(detail::usm_alloc(sycl_queue.get_context(), kind,
        kind != usm::alloc::host ? std::optional(sycl_queue.get_device()) : std::nullopt, count * sizeof(T),
        detail::least_common_alignment(alignment, alignof(T))));
}

inline void free(void *ptr, const context &sycl_context) {
    (void)sycl_context;
    detail::usm_free(ptr, sycl_context);
}

inline void free(void *ptr, const queue &sycl_queue) {
    (void)sycl_queue;
    detail::usm_free(ptr, sycl_queue.get_context());
}

usm::alloc get_pointer_type(const void *ptr, const context &sycl_context);

device get_pointer_device(const void *ptr, const context &sycl_context);

} // namespace simsycl::sycl
