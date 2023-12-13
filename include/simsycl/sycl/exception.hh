#pragma once

#include "enums.hh"
#include "forward.hh"

#include <functional>
#include <system_error>
#include <vector>

namespace simsycl::sycl {

class exception : public virtual std::exception {
  public:
    exception(std::error_code ec, const std::string &what_arg);
    exception(std::error_code ec, const char *what_arg);
    exception(std::error_code ec);
    exception(int ev, const std::error_category &ecat, const std::string &what_arg);
    exception(int ev, const std::error_category &ecat, const char *what_arg);
    exception(int ev, const std::error_category &ecat);

    exception(context ctx, std::error_code ec, const std::string &what_arg);
    exception(context ctx, std::error_code ec, const char *what_arg);
    exception(context ctx, std::error_code ec);
    exception(context ctx, int ev, const std::error_category &ecat, const std::string &what_arg);
    exception(context ctx, int ev, const std::error_category &ecat, const char *what_arg);
    exception(context ctx, int ev, const std::error_category &ecat);

    const std::error_code &code() const noexcept;
    const std::error_category &category() const noexcept;

    const char *what() const noexcept;

    bool has_context() const noexcept;
    context get_context() const;
};

class exception_list : private std::vector<std::exception_ptr> {
    // Used as a container for a list of asynchronous exceptions
  public:
    using value_type = std::exception_ptr;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = std::size_t;

    using std::vector<std::exception_ptr>::const_iterator;
    using iterator = const_iterator;

    using std::vector<std::exception_ptr>::size;
    iterator begin() const { return std::vector<std::exception_ptr>::begin(); }
    iterator end() const { return std::vector<std::exception_ptr>::end(); }
};

using async_handler = std::function<void(sycl::exception_list)>;

std::error_code make_error_code(errc e) noexcept;

const std::error_category &sycl_category() noexcept;

} // namespace simsycl::sycl

namespace std {

template<>
struct is_error_code_enum<simsycl::sycl::errc> : true_type {};

} // namespace std
