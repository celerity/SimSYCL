#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

using namespace simsycl;

template<int Dimensions>
bool check_bool_vec(sycl::vec<bool, Dimensions> a) {
    for(int i = 0; i < Dimensions; ++i) {
        if(!a[i]) { return false; }
    }
    return true;
}

TEST_CASE("Basic vector operations work as expected", "[vec]") {
    // this test has no pretentions of being exhaustive, it just instantiates a subset and does basic checks

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
