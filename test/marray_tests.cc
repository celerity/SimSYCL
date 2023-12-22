#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

using namespace simsycl;

// this test has no pretentions of being exhaustive, it just instantiates a subset and does basic checks

template<size_t NumElements>
bool check_bool_marray(sycl::marray<bool, NumElements> a) {
    for(size_t i = 0; i < NumElements; ++i) {
        if(!a[i]) { return false; }
    }
    return true;
}

TEST_CASE("Basic marraytor operations work as expected", "[marray]") {
    auto vi1 = sycl::marray<int, 1>(1);

    CHECK(check_bool_marray(vi1 + 2 == sycl::marray<int, 1>{3}));
    CHECK(check_bool_marray(-vi1 == sycl::marray<int, 1>{-1}));
    CHECK(check_bool_marray(~vi1 == sycl::marray<int, 1>{~1}));

    sycl::marray<float, 2> vf1 = {2, 3};
    sycl::marray<float, 2> vf2 = {3, 4};
    CHECK(check_bool_marray(vf1 + vf2 == sycl::marray<float, 2>{5, 7}));
    CHECK(check_bool_marray(vf1 - vf2 == sycl::marray<float, 2>{-1, -1}));
    CHECK(check_bool_marray(vf1 * vf2 == sycl::marray<float, 2>{6, 12}));
    CHECK(check_bool_marray(vf1 / vf2 == sycl::marray<float, 2>{2.0f / 3.0f, 3.0f / 4.0f}));
    CHECK(check_bool_marray(vf1 + 2 == sycl::marray<float, 2>{4, 5}));
    CHECK(check_bool_marray(vf1 - 2 == sycl::marray<float, 2>{0, 1}));
    CHECK(check_bool_marray(vf1 * 2 == sycl::marray<float, 2>{4, 6}));
    CHECK(check_bool_marray(vf1 / 2 == sycl::marray<float, 2>{1, 1.5f}));

    sycl::marray<double, 3> vd1 = {4, 5, 6};
    sycl::marray<double, 3> vd2 = {5, 6, 7};
    CHECK(check_bool_marray(vd1 + vd2 == sycl::marray<double, 3>{9, 11, 13}));
    CHECK(check_bool_marray(vd1 - vd2 == sycl::marray<double, 3>{-1, -1, -1}));
    CHECK(check_bool_marray(vd1 * vd2 == sycl::marray<double, 3>{20, 30, 42}));
    CHECK(check_bool_marray(vd1 / vd2 == sycl::marray<double, 3>{4.0 / 5.0, 5.0 / 6.0, 6.0 / 7.0}));
    CHECK(check_bool_marray(vd1 + 2 == sycl::marray<double, 3>{6, 7, 8}));
    CHECK(check_bool_marray(vd1 - 2 == sycl::marray<double, 3>{2, 3, 4}));
    CHECK(check_bool_marray(vd1 * 2 == sycl::marray<double, 3>{8, 10, 12}));
    CHECK(check_bool_marray(vd1 / 2 == sycl::marray<double, 3>{2, 2.5, 3}));

    sycl::marray<bool, 4> vb1 = {true, false, true, false};
    sycl::marray<bool, 4> vb2 = {false, true, true, false};
    CHECK(check_bool_marray((vb1 && vb2) == sycl::marray<bool, 4>{false, false, true, false}));
    CHECK(check_bool_marray((vb1 || vb2) == sycl::marray<bool, 4>{true, true, true, false}));
    CHECK(check_bool_marray((!vb1) == sycl::marray<bool, 4>{false, true, false, true}));
}

TEST_CASE("marraytor relational operators work as expected", "[marray]") {
    // this test has no pretentions of being exhaustive, it just instantiates a subset and does basic checks

    auto vi1 = sycl::marray<int, 1>{1};
    auto vi2 = sycl::marray<int, 1>{2};
    CHECK(check_bool_marray(vi1 == vi1));
    CHECK(check_bool_marray(vi1 != vi2));
    CHECK(check_bool_marray(vi1 < vi2));
    CHECK(check_bool_marray(vi1 <= vi2));
    CHECK(check_bool_marray(vi2 > vi1));
    CHECK(check_bool_marray(vi2 >= vi1));

    sycl::marray<float, 2> vf1 = {2, 3};
    sycl::marray<float, 2> vf2 = {3, 4};
    CHECK(check_bool_marray(vf1 == vf1));
    CHECK(check_bool_marray(vf1 != vf2));
    CHECK(check_bool_marray(vf1 < vf2));
    CHECK(check_bool_marray(vf1 <= vf2));
    CHECK(check_bool_marray(vf2 > vf1));
    CHECK(check_bool_marray(vf2 >= vf1));

    sycl::marray<double, 3> vd1 = {4, 5, 6};
    sycl::marray<double, 3> vd2 = {5, 6, 7};
    CHECK(check_bool_marray(vd1 == vd1));
    CHECK(check_bool_marray(vd1 != vd2));
    CHECK(check_bool_marray(vd1 < vd2));
    CHECK(check_bool_marray(vd1 <= vd2));
    CHECK(check_bool_marray(vd2 > vd1));
    CHECK(check_bool_marray(vd2 >= vd1));

    sycl::marray<bool, 4> vb1 = {true, false, true, false};
    sycl::marray<bool, 4> vb2 = {false, true, true, false};
    CHECK(check_bool_marray(vb1 == vb1));
    CHECK(check_bool_marray((vb1 != vb2) == sycl::marray<bool, 4>{true, true, false, false}));
    CHECK(check_bool_marray((vb1 < vb2) == sycl::marray<bool, 4>{false, true, false, false}));
    CHECK(check_bool_marray((vb1 <= vb2) == sycl::marray<bool, 4>{false, true, true, true}));
    CHECK(check_bool_marray((vb2 > vb1) == sycl::marray<bool, 4>{false, true, false, false}));
    CHECK(check_bool_marray((vb2 >= vb1) == sycl::marray<bool, 4>{false, true, true, true}));
}
