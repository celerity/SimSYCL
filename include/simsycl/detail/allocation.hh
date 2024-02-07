#pragma once

#include "../sycl/enums.hh"
#include "../sycl/forward.hh"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>


namespace simsycl::detail {

// aligned allocation wrapper for cross-platform support
// NOTE: returned pointers must be freed with aligned_free
inline void *aligned_alloc(size_t alignment, size_t size) {
#if defined(_MSC_VER)
    // MSVC does not have std::aligned_alloc because the pointers it returns cannot be freed with std::free
    return _aligned_malloc(size, alignment);
#else
    // POSIX and notably macOS requires a minimum alignment of sizeof(void*) for aligned_alloc, so we only use that for
    // over-aligned allocations
    if(alignment <= alignof(std::max_align_t)) {
        return std::malloc(size);
    } else {
        return std::aligned_alloc(alignment, size);
    }
#endif
}

inline void aligned_free(void *ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void *usm_alloc(const sycl::context &context, sycl::usm::alloc kind, std::optional<sycl::device> opt_device,
    size_t size_bytes, size_t alignment_bytes);
void usm_free(void *ptr, const sycl::context &context);


// floats and doubles filled with this pattern show up as "-nan"
inline constexpr std::byte uninitialized_memory_pattern = std::byte(0xff);

class allocation {
  public:
    allocation() = default;
    allocation(const size_t size_bytes, const size_t alignment_bytes)
        : m_size(size_bytes), m_alignment(alignment_bytes), m_ptr(aligned_alloc(alignment_bytes, size_bytes)) {
        memset(m_ptr, static_cast<int>(uninitialized_memory_pattern), size_bytes);
    }

    allocation(const allocation &) = delete;

    allocation(allocation &&other) noexcept {
        std::swap(m_size, other.m_size);
        std::swap(m_alignment, other.m_alignment);
        std::swap(m_ptr, other.m_ptr);
    }

    allocation &operator=(const allocation &) = delete;

    allocation &operator=(allocation &&other) noexcept {
        reset();
        std::swap(m_size, other.m_size);
        std::swap(m_alignment, other.m_alignment);
        std::swap(m_ptr, other.m_ptr);
        return *this;
    }

    ~allocation() { reset(); }

    void *get() { return m_ptr; }
    const void *get() const { return m_ptr; }

    size_t size_bytes() const { return m_size; }
    size_t alignment_bytes() const { return m_alignment; }

    void reset() {
        if(m_ptr != nullptr) {
            aligned_free(m_ptr);
            m_size = 0;
            m_alignment = 1;
            m_ptr = nullptr;
        }
    }

    explicit operator bool() const { return m_ptr != nullptr; }

    friend bool operator==(const allocation &lhs, std::nullptr_t rhs) { return lhs.m_ptr == rhs; }
    friend bool operator==(std::nullptr_t lhs, const allocation &rhs) { return lhs == rhs.m_ptr; }
    friend bool operator!=(const allocation &lhs, std::nullptr_t rhs) { return !(lhs == rhs); }
    friend bool operator!=(std::nullptr_t lhs, const allocation &rhs) { return !(lhs == rhs); }

  private:
    size_t m_size = 0;
    size_t m_alignment = 1;
    void *m_ptr = nullptr;
};

} // namespace simsycl::detail
