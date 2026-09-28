#include <catch2/catch_test_macros.hpp>

#include <cstddef>

#include <sycl/sycl.hpp>

using namespace sycl;

SIMSYCL_START_IGNORING_DEPRECATIONS

TEST_CASE("Calls to the deprecated parallel_for signature are not ambiguous", "[ambiguity][parallel_for]") {
    queue q;
    constexpr int offset = 7;
    SECTION("1D") {
        q.submit([&](handler &cgh) {
            cgh.parallel_for(range<1>{1}, id<1>{offset}, [=](id<1> i) { CHECK(i[0] == offset); });
        });
    }
    SECTION("2D") {
        q.submit([&](handler &cgh) {
            cgh.parallel_for(range<2>{1, 1}, id<2>{0, offset}, [=](id<2> i) { CHECK(i == id<2>{0, offset}); });
        });
    }
    SECTION("3D") {
        q.submit([&](handler &cgh) {
            cgh.parallel_for(range<3>{1, 1, 1}, id<3>{0, offset, 0}, [=](id<3> i) { CHECK(i == id<3>{0, offset, 0}); });
        });
    }
}

SIMSYCL_STOP_IGNORING_DEPRECATIONS

TEST_CASE("Calls of operator op with integral type are not abiguous", "[ambiguity]") {
    SECTION("size_t") {
        [[maybe_unused]] auto multiply = [](id<1> i, size_t j) {
            return i * j;
	};
    }
    SECTION("int") {
        [[maybe_unused]] auto multiply = [](id<1> i, int j) {
            return i * j;
        };
    }
}
