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
    half h = static_cast<half>(7.0f);
    auto vh1 = vec<half, 2>(half(3.0), half(4.0));
    CHECK(length(h) == Catch::Approx(7.0f));
    CHECK(length(vh1) == Catch::Approx(5.0f));
    CHECK(length(vh1.yx()) == Catch::Approx(5.0f));
#endif
}

TEST_CASE("Dot function works as expected", "[math][geometric]") {
    float x = 8.0f;
    double y = 7.0f;
    CHECK(dot(x, x) == Catch::Approx(64.0f));
    CHECK(dot(y, y) == Catch::Approx(49.0f));


    vec<double, 2> v1 = {1.0, 2.0};
    vec<float, 2> v2 = {3.0f, 4.0f};
    vec<double, 3> v3 = {5.0, 6.0, 7.0};
    vec<float, 3> v4 = {8.0f, 9.0f, 10.0f};
    vec<double, 4> v5 = {11.0, 12.0, 13.0, 14.0};
    vec<float, 4> v6 = {15.0f, 16.0f, 17.0f, 18.0f};

    CHECK(dot(v1, v1) == Catch::Approx(5.0));
    CHECK(dot(v2, v2) == Catch::Approx(25.0f));
    CHECK(dot(v3, v3) == Catch::Approx(110.0));
    CHECK(dot(v4, v4) == Catch::Approx(245.0f));
    CHECK(dot(v5, v5) == Catch::Approx(630.0));
    CHECK(dot(v6, v6) == Catch::Approx(1094.0f));
    CHECK(dot(v1.xy(), v1.yx()) == Catch::Approx(4.0));
    CHECK(dot(v2.xx(), v2.xy()) == Catch::Approx(21.0f));
    CHECK(dot(v6.argb(), v2.xyxy()) == Catch::Approx(230.0f));
    CHECK(dot(v3, v1.xyx()) == Catch::Approx(24.0));

    marray<float, 4> m1 = {1.0f, 2.0f, 3.0f, 4.0f};
    marray<float, 4> m2 = {5.0f, 6.0f, 7.0f, 8.0f};
    CHECK(dot(m1, m2) == Catch::Approx(70.0f));

#if SIMSYCL_FEATURE_HALF_TYPE
    using sycl::half;
    auto vh1 = vec<half, 2>(half(1.0), half(2.0));
    auto vh2 = vec<half, 2>(half(3.0), half(4.0));
    CHECK(dot(vh1, vh1) == Catch::Approx(5.0f));
    CHECK(dot(vh2, vh2) == Catch::Approx(25.0f));
    CHECK(dot(vh1, vh2) == Catch::Approx(11.0f));
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

TEST_CASE("Normalize function works as expected", "[math][geometric]") {
    double x = 8.0f;
    float y = 7.0f;
    vec<double, 2> v1 = {1.0, 2.0};
    vec<float, 2> v2 = {3.0f, 4.0f};
    vec<double, 3> v3 = {5.0, 6.0, 7.0};
    vec<float, 3> v4 = {8.0f, 9.0f, 10.0f};
    vec<double, 4> v5 = {11.0, 12.0, 13.0, 14.0};
    vec<float, 4> v6 = {15.0f, 16.0f, 17.0f, 18.0f};

    CHECK(normalize(x) == Catch::Approx(1.0));
    CHECK(normalize(y) == Catch::Approx(1.0f));
    auto v1n = normalize(v1);
    CHECK(v1n.x() == Catch::Approx(0.4472135954999579));
    CHECK(v1n.y() == Catch::Approx(0.8944271909999159));
    auto v1xxn = normalize(v1.xx());
    CHECK(v1xxn.x() == Catch::Approx(0.7071067811865475));
    CHECK(v1xxn.y() == Catch::Approx(0.7071067811865475));
    auto v2n = normalize(v2);
    CHECK(v2n.x() == Catch::Approx(0.6f));
    CHECK(v2n.y() == Catch::Approx(0.8f));
    auto v2yxn = normalize(v2.yx());
    CHECK(v2yxn.x() == Catch::Approx(0.8f));
    CHECK(v2yxn.y() == Catch::Approx(0.6f));
    auto v3n = normalize(v3);
    CHECK(v3n.x() == Catch::Approx(0.4767312946227962));
    CHECK(v3n.y() == Catch::Approx(0.5720775535473553));
    CHECK(v3n.z() == Catch::Approx(0.6674238124719146));
    auto v4n = normalize(v4);
    CHECK(v4n.x() == Catch::Approx(0.5111012519999519f));
    CHECK(v4n.y() == Catch::Approx(0.5749889084999459f));
    CHECK(v4n.z() == Catch::Approx(0.6388765649999398f));
    auto v5n = normalize(v5);
    CHECK(v5n.x() == Catch::Approx(0.4382504900892777));
    CHECK(v5n.y() == Catch::Approx(0.4780914437337575));
    CHECK(v5n.z() == Catch::Approx(0.5179323973782373));
    CHECK(v5n.w() == Catch::Approx(0.5577733510227171));
    auto v6n = normalize(v6);
    CHECK(v6n.r() == Catch::Approx(0.4535055413676754f));
    CHECK(v6n.g() == Catch::Approx(0.4837392441255204f));
    CHECK(v6n.b() == Catch::Approx(0.5139729468833655f));
    CHECK(v6n.a() == Catch::Approx(0.5442066496412105f));
    auto v6argnn = normalize(v6.argb());
    CHECK(v6argnn.x() == Catch::Approx(0.5442066496412105f));
    CHECK(v6argnn.y() == Catch::Approx(0.4535055413676754f));
    CHECK(v6argnn.z() == Catch::Approx(0.4837392441255204f));
    CHECK(v6argnn.w() == Catch::Approx(0.5139729468833655f));
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

TEST_CASE("Inverse square root function works as expected", "[math]") {
    CHECK(rsqrt(8.0f) == Catch::Approx(1.0 / sqrt(8.0f)));
    CHECK(rsqrt(7.0) == Catch::Approx(1.0 / sqrt(7.0)));

    vec<double, 2> v1 = {2.0, 9.0};
    vec<double, 2> v1_result = rsqrt(v1);
    CHECK(v1_result.x() == Catch::Approx(rsqrt(v1.x())));
    CHECK(v1_result.y() == Catch::Approx(rsqrt(v1.y())));
}

TEST_CASE("Popcount function works as expected", "[math]") {
    CHECK(popcount(static_cast<std::uint8_t>(-1)) == 8);
    CHECK(popcount(static_cast<std::uint16_t>(-1)) == 16);
    CHECK(popcount(static_cast<std::uint32_t>(-1)) == 32);
    CHECK(popcount(static_cast<std::uint64_t>(-1)) == 64);

    CHECK(popcount(static_cast<signed char>(0b101010)) == 3);
    CHECK(popcount(static_cast<unsigned char>(0b111111)) == 6);
    CHECK(popcount(static_cast<signed short>(0b101010)) == 3);
    CHECK(popcount(static_cast<unsigned short>(0b111111)) == 6);
    CHECK(popcount(static_cast<signed int>(0b101010)) == 3);
    CHECK(popcount(static_cast<unsigned int>(0b111111)) == 6);
    CHECK(popcount(static_cast<signed long>(0b101010)) == 3);
    CHECK(popcount(static_cast<unsigned long>(0b111111)) == 6);
    CHECK(popcount(static_cast<signed long long>(0b101010)) == 3);
    CHECK(popcount(static_cast<unsigned long long>(0b111111)) == 6);

    vec<int, 4> v1 = {0b101010, 0b111111, 0b101010, -2};
    vec<int, 4> v1_result = popcount(v1);
    CHECK(v1_result.x() == 3);
    CHECK(v1_result.y() == 6);
    CHECK(v1_result.z() == 3);
    CHECK(v1_result.w() == 31);

    vec<std::uint8_t, 3> v2 = {0b0, 0b1101, 0b1111};
    auto v2_result = popcount(v2.xxyz());
    CHECK(v2_result.x() == 0);
    CHECK(v2_result.y() == 0);
    CHECK(v2_result.z() == 3);
    CHECK(v2_result.w() == 4);
}
