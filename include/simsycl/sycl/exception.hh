#pragma once

#include "context.hh"
#include "enums.hh"

#include <optional>
#include <system_error>


namespace simsycl::sycl {

class exception : public virtual std::exception {
  public:
    exception(std::error_code ec, const std::string &what_arg) : m_error(ec, what_arg){};
    exception(std::error_code ec, const char *what_arg) : m_error(ec, what_arg){};
    exception(std::error_code ec) : m_error(ec){};
    exception(int ev, const std::error_category &ecat, const std::string &what_arg) : m_error(ev, ecat, what_arg){};
    exception(int ev, const std::error_category &ecat, const char *what_arg) : m_error(ev, ecat, what_arg){};
    exception(int ev, const std::error_category &ecat) : m_error(ev, ecat){};

    exception(context ctx, std::error_code ec, const std::string &what_arg) : m_error(ec, what_arg), m_context(ctx){};
    exception(context ctx, std::error_code ec, const char *what_arg) : m_error(ec, what_arg), m_context(ctx){};
    exception(context ctx, std::error_code ec) : m_error(ec), m_context(ctx){};
    exception(context ctx, int ev, const std::error_category &ecat, const std::string &what_arg)
        : m_error(ev, ecat, what_arg), m_context(ctx){};
    exception(context ctx, int ev, const std::error_category &ecat, const char *what_arg)
        : m_error(ev, ecat, what_arg), m_context(ctx){};
    exception(context ctx, int ev, const std::error_category &ecat) : m_error(ev, ecat), m_context(ctx){};

    const std::error_code &code() const noexcept { return m_error.code(); }
    const std::error_category &category() const noexcept { return m_error.code().category(); }

    const char *what() const noexcept override { return m_error.what(); }

    bool has_context() const noexcept { return m_context.has_value(); }
    context get_context() const { return m_context.value(); }

  private:
    std::system_error m_error;
    std::optional<context> m_context;
};

std::error_code make_error_code(errc e) noexcept;

const std::error_category &sycl_category() noexcept;

} // namespace simsycl::sycl

namespace std {

template<>
struct is_error_code_enum<simsycl::sycl::errc> : true_type {};

} // namespace std
