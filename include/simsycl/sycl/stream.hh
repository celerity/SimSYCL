// Dear god, why?

#pragma once

#include "../detail/reference_type.hh"
#include "../sycl/enums.hh"
#include "../sycl/forward.hh"
#include "../sycl/property.hh"

namespace simsycl::detail {

class precision_manipulator;
class width_manipulator;

struct stream_state {
    size_t total_buffer_size;
    size_t work_item_buffer_size;

    stream_state(const size_t total_buffer_size, const size_t work_item_buffer_size)
        : total_buffer_size(total_buffer_size), work_item_buffer_size(work_item_buffer_size) {}
};

} // namespace simsycl::detail

namespace simsycl::sycl {

inline constexpr stream_manipulator flush = stream_manipulator::flush;
inline constexpr stream_manipulator dec = stream_manipulator::dec;
inline constexpr stream_manipulator hex = stream_manipulator::hex;
inline constexpr stream_manipulator oct = stream_manipulator::oct;
inline constexpr stream_manipulator noshowbase = stream_manipulator::noshowbase;
inline constexpr stream_manipulator showbase = stream_manipulator::showbase;
inline constexpr stream_manipulator noshowpos = stream_manipulator::noshowpos;
inline constexpr stream_manipulator showpos = stream_manipulator::showpos;
inline constexpr stream_manipulator endl = stream_manipulator::endl;
inline constexpr stream_manipulator fixed = stream_manipulator::fixed;
inline constexpr stream_manipulator scientific = stream_manipulator::scientific;
inline constexpr stream_manipulator hexfloat = stream_manipulator::hexfloat;
inline constexpr stream_manipulator defaultfloat = stream_manipulator::defaultfloat;

detail::precision_manipulator setprecision(int precision);
detail::width_manipulator setw(int width);

class stream : public detail::reference_type<stream, detail::stream_state>, public detail::property_interface {
  private:
    using reference_type = detail::reference_type<stream, detail::stream_state>;
    using property_compatibility = detail::property_compatibility_with<stream>;

  public:
    stream(size_t total_buffer_size, size_t work_item_buffer_size, handler &cgh, const property_list &prop_list = {})
        : reference_type(std::in_place, total_buffer_size, work_item_buffer_size),
          detail::property_interface(prop_list, property_compatibility()) {
        (void)cgh;
    }

    size_t size() const noexcept { return state().total_buffer_size; }

    [[deprecated]] size_t get_size() const { return size(); }

    size_t get_work_item_buffer_size() const { return state().work_item_buffer_size; }

    [[deprecated]] size_t get_max_statement_size() const { return get_work_item_buffer_size(); }
};

template<typename T>
const stream &operator<<(const stream &os, const T &rhs);

} // namespace simsycl::sycl

template<>
struct std::hash<simsycl::sycl::stream> {
    size_t operator()(const simsycl::sycl::stream &s) const noexcept {
        return std::hash<simsycl::detail::reference_type<simsycl::sycl::stream, simsycl::detail::stream_state>>{}(s);
    }
};

namespace simsycl::detail {

class precision_manipulator {
  private:
    friend precision_manipulator sycl::setprecision(int precision);

    explicit precision_manipulator(int precision) : m_precision(precision) {}
    int get_precision() const { return m_precision; }

    int m_precision;
};

class width_manipulator {
  private:
    friend width_manipulator sycl::setw(int width);

    explicit width_manipulator(int width) : m_width(width) {}
    int get_width() const { return m_width; }

    int m_width;
};

} // namespace simsycl::detail
