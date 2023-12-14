#pragma once

#include <exception>
#include <functional>
#include <vector>


namespace simsycl::sycl {

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

}

namespace simsycl::detail {

[[noreturn]] void default_async_handler(sycl::exception_list exceptions);

void call_async_handler(const sycl::async_handler &handler_opt, sycl::exception_list exceptions);

}
