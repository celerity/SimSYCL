#include <catch2/catch_test_macros.hpp>

#define SYCL_SIMPLE_SWIZZLES
#include <sycl/sycl.hpp>

#include "test_utils.hh"

using namespace simsycl;
using simsycl::test::check_bool_vec;

// this test has no pretentions of being exhaustive, it just instantiates a subset and does basic checks

TEST_CASE("Compile time vector operations work as expected", "[vec]") {
    CHECK(detail::generic_num_elements_v<float> == 1);
    CHECK(detail::generic_num_elements_v<sycl::vec<float, 1>> == 1);
    CHECK(detail::generic_num_elements_v<sycl::vec<double, 2>> == 2);
    CHECK(detail::generic_num_elements_v<sycl::vec<int, 3>> == 3);
    CHECK(detail::generic_num_elements_v<sycl::vec<float, 4>> == 4);
}

TEST_CASE("Basic vector operations work as expected", "[vec]") {
    auto vi1 = sycl::vec<int, 1>(1);

    CHECK(check_bool_vec(vi1 + 2 == sycl::vec<int, 1>{3}));
    CHECK(check_bool_vec(-vi1 == sycl::vec<int, 1>{-1}));
    CHECK(check_bool_vec(~vi1 == sycl::vec<int, 1>{~1}));

    sycl::vec<float, 2> vf1 = {2, 3};
    sycl::vec<float, 2> vf2 = {3, 4};
    CHECK(check_bool_vec(vf1 + vf2 == sycl::vec<float, 2>{5, 7}));
    CHECK(check_bool_vec(vf1 - vf2 == sycl::vec<float, 2>{-1, -1}));
    CHECK(check_bool_vec(vf1 * vf2 == sycl::vec<float, 2>{6, 12}));
    CHECK(check_bool_vec(vf1 / vf2 == sycl::vec<float, 2>{2.0f / 3.0f, 3.0f / 4.0f}));
    CHECK(check_bool_vec(vf1 + 2 == sycl::vec<float, 2>{4, 5}));
    CHECK(check_bool_vec(vf1 - 2 == sycl::vec<float, 2>{0, 1}));
    CHECK(check_bool_vec(vf1 * 2 == sycl::vec<float, 2>{4, 6}));
    CHECK(check_bool_vec(vf1 / 2 == sycl::vec<float, 2>{1, 1.5f}));

    sycl::vec<double, 3> vd1 = {4, 5, 6};
    sycl::vec<double, 3> vd2 = {5, 6, 7};
    CHECK(check_bool_vec(vd1 + vd2 == sycl::vec<double, 3>{9, 11, 13}));
    CHECK(check_bool_vec(vd1 - vd2 == sycl::vec<double, 3>{-1, -1, -1}));
    CHECK(check_bool_vec(vd1 * vd2 == sycl::vec<double, 3>{20, 30, 42}));
    CHECK(check_bool_vec(vd1 / vd2 == sycl::vec<double, 3>{4.0 / 5.0, 5.0 / 6.0, 6.0 / 7.0}));
    CHECK(check_bool_vec(vd1 + 2 == sycl::vec<double, 3>{6, 7, 8}));
    CHECK(check_bool_vec(vd1 - 2 == sycl::vec<double, 3>{2, 3, 4}));
    CHECK(check_bool_vec(vd1 * 2 == sycl::vec<double, 3>{8, 10, 12}));
    CHECK(check_bool_vec(vd1 / 2 == sycl::vec<double, 3>{2, 2.5, 3}));

    sycl::vec<bool, 4> vb1 = {true, false, true, false};
    sycl::vec<bool, 4> vb2 = {false, true, true, false};
    CHECK(check_bool_vec((vb1 && vb2) == sycl::vec<bool, 4>{false, false, true, false}));
    CHECK(check_bool_vec((vb1 || vb2) == sycl::vec<bool, 4>{true, true, true, false}));
    CHECK(check_bool_vec((!vb1) == sycl::vec<bool, 4>{false, true, false, true}));
}

TEST_CASE("Vector relational operators work as expected", "[vec]") {
    // this test has no pretentions of being exhaustive, it just instantiates a subset and does basic checks

    auto vi1 = sycl::vec<int, 1>{1};
    auto vi2 = sycl::vec<int, 1>{2};
    CHECK(check_bool_vec(vi1 == vi1));
    CHECK(check_bool_vec(vi1 != vi2));
    CHECK(check_bool_vec(vi1 < vi2));
    CHECK(check_bool_vec(vi1 <= vi2));
    CHECK(check_bool_vec(vi2 > vi1));
    CHECK(check_bool_vec(vi2 >= vi1));

    sycl::vec<float, 2> vf1 = {2, 3};
    sycl::vec<float, 2> vf2 = {3, 4};
    CHECK(check_bool_vec(vf1 == vf1));
    CHECK(check_bool_vec(vf1 != vf2));
    CHECK(check_bool_vec(vf1 < vf2));
    CHECK(check_bool_vec(vf1 <= vf2));
    CHECK(check_bool_vec(vf2 > vf1));
    CHECK(check_bool_vec(vf2 >= vf1));

    sycl::vec<double, 3> vd1 = {4, 5, 6};
    sycl::vec<double, 3> vd2 = {5, 6, 7};
    CHECK(check_bool_vec(vd1 == vd1));
    CHECK(check_bool_vec(vd1 != vd2));
    CHECK(check_bool_vec(vd1 < vd2));
    CHECK(check_bool_vec(vd1 <= vd2));
    CHECK(check_bool_vec(vd2 > vd1));
    CHECK(check_bool_vec(vd2 >= vd1));

    sycl::vec<bool, 4> vb1 = {true, false, true, false};
    sycl::vec<bool, 4> vb2 = {false, true, true, false};
    CHECK(check_bool_vec(vb1 == vb1));
    CHECK(check_bool_vec((vb1 != vb2) == sycl::vec<bool, 4>{true, true, false, false}));
    CHECK(check_bool_vec((vb1 < vb2) == sycl::vec<bool, 4>{false, true, false, false}));
    CHECK(check_bool_vec((vb1 <= vb2) == sycl::vec<bool, 4>{false, true, true, true}));
    CHECK(check_bool_vec((vb2 > vb1) == sycl::vec<bool, 4>{false, true, false, false}));
    CHECK(check_bool_vec((vb2 >= vb1) == sycl::vec<bool, 4>{false, true, true, true}));
}

TEST_CASE("Vector swizzled access is available", "[vec][swizzle]") {
    SECTION("2 components") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        CHECK(check_bool_vec(vi.xx() == sycl::vec<int, 2>{1, 1}));
        vi.xy() = {7, 8};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{7, 8, 3, 4}));
        vi.wz() = {9, 10};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{7, 8, 10, 9}));

        sycl::vec<float, 4> vf = {1, 2, 3, 4};
        vf.gr() = {4, 5};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{5, 4, 3, 4}));
        vf.bg() = {6, 7};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{5, 7, 6, 4}));
    }

    SECTION("3 components") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        CHECK(check_bool_vec(vi.xxzz() == sycl::vec<int, 4>{1, 1, 3, 3}));
        vi.xyz() = {5, 6, 7};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{5, 6, 7, 4}));
        vi.wyx() = {8, 9, 10};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{10, 9, 7, 8}));
        vi.z() = {11};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{10, 9, 11, 8}));

        sycl::vec<float, 4> vf = {1, 2, 3, 4};
        vf.bgr() = {4, 5, 6};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{6, 5, 4, 4}));
        vf.rgb() = {7, 8, 9};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{7, 8, 9, 4}));
        vf.g() = {10};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{7, 10, 9, 4}));
    }

    SECTION("4 components") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        vi.xyzw() = {5, 6, 7, 8};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{5, 6, 7, 8}));
        vi.wzyx() = {9, 10, 11, 12};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{12, 11, 10, 9}));
        vi.z() = {13};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{12, 11, 13, 9}));

        sycl::vec<float, 4> vf = {1, 2, 3, 4};
        vf.bgra() = {4, 5, 6, 7};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{6, 5, 4, 7}));
        vf.rgba() = {8, 9, 10, 11};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{8, 9, 10, 11}));
        vf.a() = {12};
        CHECK(check_bool_vec(vf == sycl::vec<float, 4>{8, 9, 10, 12}));
    }

    SECTION("lo, hi, odd and even") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        vi.lo() = {5, 6};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{5, 6, 3, 4}));
        vi.hi() = {7, 8};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{5, 6, 7, 8}));
        vi.odd() = {9, 10};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{5, 9, 7, 10}));
        vi.even() = {11, 12};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{11, 9, 12, 10}));
    }
}

TEST_CASE("Operations on swizzled vectors work as expected", "[vec][swizzle]") {
    SECTION("binary operators") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        sycl::vec<int, 2> vi2 = {5, 6};
        CHECK(check_bool_vec(vi.xx() + vi.zw() == sycl::vec<int, 2>{4, 5}));
        CHECK(check_bool_vec(vi.zz() + vi2 == sycl::vec<int, 2>{8, 9}));
        CHECK(check_bool_vec(vi2 + vi.ww() == sycl::vec<int, 2>{9, 10}));
        CHECK(check_bool_vec(1 + vi.yz() == sycl::vec<int, 2>{3, 4}));
        CHECK(check_bool_vec(vi.xxz() + 1 == sycl::vec<int, 3>{2, 2, 4}));
    }

    SECTION("compound operators") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        vi.xy() += 1;
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{2, 3, 3, 4}));
        vi.zw() += sycl::vec<int, 2>{5, 6};
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{2, 3, 8, 10}));
        vi.ra() -= vi.ra();
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{0, 3, 8, 0}));
    }

    SECTION("unary operators") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        CHECK(check_bool_vec(-vi.xy() == sycl::vec<int, 2>{-1, -2}));
        CHECK(check_bool_vec(~vi.zw() == sycl::vec<int, 2>{~3, ~4}));

        CHECK(check_bool_vec(++vi.xz() == sycl::vec<int, 2>{2, 4}));
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{2, 2, 4, 4}));
        CHECK(check_bool_vec(vi.xyw()-- == sycl::vec<int, 3>{2, 2, 4}));
        CHECK(check_bool_vec(vi == sycl::vec<int, 4>{1, 1, 4, 3}));

        sycl::vec<bool, 3> vb = {true, false, true};
        CHECK(check_bool_vec((!vb.yz()) == sycl::vec<bool, 2>{true, false}));
    }

    SECTION("comparison operators") {
        sycl::vec<int, 4> vi = {1, 2, 3, 4};
        CHECK(check_bool_vec(vi.xy() == vi.xy()));
        CHECK(check_bool_vec(vi.xy() != vi.zw()));
        CHECK(check_bool_vec(vi.xy() < vi.zw()));
        CHECK(check_bool_vec(vi.xy() <= vi.zw()));
        CHECK(check_bool_vec(vi.zw() > vi.xy()));
        CHECK(check_bool_vec(vi.zw() >= vi.xy()));

        CHECK(check_bool_vec(vi.yx() == sycl::vec<int, 2>{2, 1}));
        CHECK(check_bool_vec(sycl::vec<int, 2>{0, 0} < vi.xy()));

        CHECK(check_bool_vec(vi.zw() < 10));
        CHECK(check_bool_vec(vi.z() == 3));
        CHECK(check_bool_vec(4 == vi.a()));
    }
}
