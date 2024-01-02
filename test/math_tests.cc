#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#define SYCL_SIMPLE_SWIZZLES
#include <sycl/sycl.hpp>

#include "test_utils.hh"

using namespace sycl;

TEST_CASE("Length function works as expected", "[math][geometric]") {
    double x = 8.0f;
    float y = 7.0f;
    vec<double, 2> v1 = {1.0, 2.0};
    vec<float, 2> v2 = {3.0f, 4.0f};
    vec<double, 3> v3 = {5.0, 6.0, 7.0};
    vec<float, 3> v4 = {8.0f, 9.0f, 10.0f};
    vec<double, 4> v5 = {11.0, 12.0, 13.0, 14.0};
    vec<float, 4> v6 = {15.0f, 16.0f, 17.0f, 18.0f};

    CHECK(length(x) == Catch::Approx(8.0));
    CHECK(length(y) == Catch::Approx(7.0f));
    CHECK(length(v1) == Catch::Approx(2.23606797749979));
    CHECK(length(v2) == Catch::Approx(5.0f));
    CHECK(length(v3) == Catch::Approx(10.488088481701515));
    CHECK(length(v4) == Catch::Approx(15.6524758425f));
    CHECK(length(v5) == Catch::Approx(25.099800796));
    CHECK(length(v6) == Catch::Approx(33.07567f));
    CHECK(length(v1.xx()) == Catch::Approx(1.4142135624));
    CHECK(length(v6.argb()) == Catch::Approx(33.07567f));

    marray<float, 4> m1 = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(length(m1) == Catch::Approx(5.4772255751f));

#if SIMSYCL_FEATURE_HALF_TYPE
    using sycl::half;
    half h = 7.0f;
    auto vh1 = vec<half, 2>(half(3.0), half(4.0));
    CHECK(length(h) == Catch::Approx(7.0f));
    CHECK(length(vh1) == Catch::Approx(5.0f));
    CHECK(length(vh1.yx()) == Catch::Approx(5.0f));
#endif
}

TEST_CASE("Distance function works as expected", "[math][geometric]") {
    double x = 8.0f;
    float y = 7.0f;
    vec<double, 2> v1 = {1.0, 2.0};
    vec<float, 2> v2 = {3.0f, 4.0f};
    vec<double, 3> v3 = {5.0, 6.0, 7.0};
    vec<float, 3> v4 = {8.0f, 9.0f, 10.0f};
    vec<double, 4> v5 = {11.0, 12.0, 13.0, 14.0};
    vec<float, 4> v6 = {15.0f, 16.0f, 17.0f, 18.0f};

    CHECK(distance(x, 2.0) == Catch::Approx(6.0));
    CHECK(distance(y, -1.0f) == Catch::Approx(8.0f));
    CHECK(distance(v1, vec<double, 2>{0.0, 0.0}) == Catch::Approx(2.23606797749979));
    CHECK(distance(v2, vec<float, 2>{0.0f, 0.0f}) == Catch::Approx(5.0f));
    CHECK(distance(v3, vec<double, 3>{0.0, 0.0, 0.0}) == Catch::Approx(10.488088481701515));
    CHECK(distance(v4, vec<float, 3>{0.0f, 0.0f, 0.0f}) == Catch::Approx(15.6524758425f));
    CHECK(distance(v5, vec<double, 4>{0.0, 0.0, 0.0, 0.0}) == Catch::Approx(25.099800796));
    CHECK(distance(v6, vec<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}) == Catch::Approx(33.07567f));
    CHECK(distance(v1.xx(), v1.yx()) == Catch::Approx(1.0));
    CHECK(distance(v6.argb(), vec<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}) == Catch::Approx(33.07567f));
    CHECK(distance(v6.argb(), v2.xyxy()) == Catch::Approx(26.15339f));
}

TEST_CASE("Clamp function works as expected", "[math]") {
    using simsycl::test::check_bool_vec;

    int x = 8;
    CHECK(clamp(x, 0, 10) == 8);
    CHECK(clamp(x, 0, 5) == 5);
    CHECK(clamp(x, 10, 20) == 10);

    vec<int, 4> v1 = {1, 2, 3, 4};
    CHECK(check_bool_vec(clamp(v1, v1, v1) == v1));
    CHECK(check_bool_vec(clamp(v1, 0, 10) == v1));
    CHECK(check_bool_vec(clamp(v1, v1.gggg(), v1.bbbb()) == v1.ggbb()));

    vec<float, 4> v2 = {1.f, 2.f, 3.f, 4.f};
    CHECK(check_bool_vec(clamp(v2, v2, v2) == v2));
    CHECK(check_bool_vec(clamp(v2, 0, 10) == v2));
    CHECK(check_bool_vec(clamp(v2, v2.zzzz(), v2.wwww()) == v2.zzzw()));
}