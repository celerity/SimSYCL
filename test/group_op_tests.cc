#include "simsycl/sycl/group_algorithms.hh"
#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

using namespace simsycl;

template<sycl::Group G>
void check_group_op_sequence(const G &g, const std::vector<detail::group_operation_id> &expected_ids) {
    CHECK(detail::get_group_impl(g).operations.size() == expected_ids.size());
    for(size_t i = 0; i < expected_ids.size(); ++i) {
        CHECK(detail::get_group_impl(g).operations[i].id == expected_ids[i]);
    }
}

TEST_CASE("Group barriers behave as expected", "[group_op]") {
    enum class point { a, b, c };
    struct record {
        point p;
        size_t global_id;
        size_t global_range;
        size_t group_id;
        size_t local_id;
        size_t local_range;
        auto operator<=>(const record &) const = default;
    };
    const std::vector<record> expected = {
        {point::a, 0, 4, 0, 0, 2},
        {point::a, 1, 4, 0, 1, 2},
        {point::a, 2, 4, 1, 0, 2},
        {point::a, 3, 4, 1, 1, 2},
        {point::b, 0, 4, 0, 0, 2},
        {point::b, 1, 4, 0, 1, 2},
        {point::b, 2, 4, 1, 0, 2},
        {point::b, 3, 4, 1, 1, 2},
        {point::c, 0, 4, 0, 0, 2},
        {point::c, 1, 4, 0, 1, 2},
        {point::c, 2, 4, 1, 0, 2},
        {point::c, 3, 4, 1, 1, 2},
    };
    std::vector<record> actual;

    SECTION("For work groups") {
        sycl::queue().submit([&actual](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{4, 2}, [&actual](sycl::nd_item<1> it) {
                actual.emplace_back(point::a, it.get_global_id(0), it.get_global_range(0), it.get_group(0),
                    it.get_local_id(0), it.get_local_range(0));
                sycl::group_barrier(it.get_group());
                actual.emplace_back(point::b, it.get_global_id(0), it.get_global_range(0), it.get_group(0),
                    it.get_local_id(0), it.get_local_range(0));
                sycl::group_barrier(it.get_group());
                actual.emplace_back(point::c, it.get_global_id(0), it.get_global_range(0), it.get_group(0),
                    it.get_local_id(0), it.get_local_range(0));

                check_group_op_sequence(
                    it.get_group(), {detail::group_operation_id::barrier, detail::group_operation_id::barrier});
            });
        });
        CHECK(actual == expected);
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 2u};
        sycl::queue().submit([&actual](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{4, 4}, [&actual](sycl::nd_item<1> it) {
                const auto &sg = it.get_sub_group();
                actual.emplace_back(point::a, it.get_global_id(0), it.get_global_range(0), sg.get_group_linear_id(),
                    sg.get_local_linear_id(), sg.get_local_range().size());
                sycl::group_barrier(sg);
                actual.emplace_back(point::b, it.get_global_id(0), it.get_global_range(0), sg.get_group_linear_id(),
                    sg.get_local_linear_id(), sg.get_local_range().size());
                sycl::group_barrier(sg);
                actual.emplace_back(point::c, it.get_global_id(0), it.get_global_range(0), sg.get_group_linear_id(),
                    sg.get_local_linear_id(), sg.get_local_range().size());

                check_group_op_sequence(
                    it.get_sub_group(), {detail::group_operation_id::barrier, detail::group_operation_id::barrier});
            });
        });
        CHECK(actual == expected);
    }
}

TEST_CASE("Group broadcasts behave as expected", "[group_op][broadcast]") {
    std::vector<int> expected = {42, 42, 42, 42, 46, 46, 46, 46};
    std::vector<int> actual(expected.size());

    SECTION("For work groups") {
        sycl::queue().submit([&actual](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&actual](sycl::nd_item<1> it) {
                actual[it.get_global_linear_id()]
                    = sycl::group_broadcast(it.get_group(), 40 + it.get_global_linear_id(), 2);

                check_group_op_sequence(it.get_group(), {detail::group_operation_id::broadcast});
            });
        });
        CHECK(actual == expected);

        // default-0-id signature variant
        sycl::queue().submit([](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [](sycl::nd_item<1> it) {
                CHECK(sycl::group_broadcast(it.get_group(), 40 + it.get_global_linear_id()) //
                    == 40 + it.get_group_linear_id() * 4);
                check_group_op_sequence(it.get_group(), {detail::group_operation_id::broadcast});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&actual](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&actual](sycl::nd_item<1> it) {
                actual[it.get_global_linear_id()]
                    = sycl::group_broadcast(it.get_sub_group(), 40 + it.get_global_linear_id(), 2);

                check_group_op_sequence(it.get_sub_group(), {detail::group_operation_id::broadcast});
            });
        });
        CHECK(actual == expected);
    }
}


TEST_CASE("Group joint_any_of behaves as expected", "[group_op][joint_any_of]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_any_of(it.get_group(), inputs, inputs + 4, [](int i) { return i == 3; }));
                CHECK_FALSE(sycl::joint_any_of(it.get_group(), inputs, inputs + 4, [](int i) { return i == 5; }));
                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::joint_any_of, detail::group_operation_id::joint_any_of});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_any_of(it.get_sub_group(), inputs, inputs + 4, [](int i) { return i == 3; }));
                CHECK_FALSE(sycl::joint_any_of(it.get_sub_group(), inputs, inputs + 4, [](int i) { return i == 5; }));
                check_group_op_sequence(it.get_sub_group(),
                    {detail::group_operation_id::joint_any_of, detail::group_operation_id::joint_any_of});
            });
        });
    }
}

TEST_CASE("Group any_of_group behaves as expected", "[group_op][any_of_group]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_local_linear_id();
                CHECK(sycl::any_of_group(it.get_group(), inputs[id], [](int i) { return i == 3; }));
                CHECK_FALSE(sycl::any_of_group(it.get_group(), inputs[id], [](int i) { return i == 5; }));
                check_group_op_sequence(
                    it.get_group(), {detail::group_operation_id::any_of, detail::group_operation_id::any_of});
            });
        });

        // bool-only signature variant
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_local_linear_id();
                CHECK(sycl::any_of_group(it.get_group(), inputs[id] == 3));
                CHECK_FALSE(sycl::any_of_group(it.get_group(), inputs[id] == 5));
                check_group_op_sequence(
                    it.get_group(), {detail::group_operation_id::any_of, detail::group_operation_id::any_of});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_sub_group().get_local_linear_id();
                CHECK(sycl::any_of_group(it.get_sub_group(), inputs[id], [](int i) { return i == 3; }));
                CHECK_FALSE(sycl::any_of_group(it.get_sub_group(), inputs[id], [](int i) { return i == 5; }));
                check_group_op_sequence(
                    it.get_sub_group(), {detail::group_operation_id::any_of, detail::group_operation_id::any_of});
            });
        });
    }
}

TEST_CASE("Group joint_all_of behaves as expected", "[group_op][joint_all_of]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_all_of(it.get_group(), inputs, inputs + 4, [](int i) { return i <= 4; }));
                CHECK_FALSE(sycl::joint_all_of(it.get_group(), inputs, inputs + 4, [](int i) { return i < 4; }));
                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::joint_all_of, detail::group_operation_id::joint_all_of});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_all_of(it.get_sub_group(), inputs, inputs + 4, [](int i) { return i <= 4; }));
                CHECK_FALSE(sycl::joint_all_of(it.get_sub_group(), inputs, inputs + 4, [](int i) { return i < 4; }));
                check_group_op_sequence(it.get_sub_group(),
                    {detail::group_operation_id::joint_all_of, detail::group_operation_id::joint_all_of});
            });
        });
    }
}

TEST_CASE("Group all_of_group behaves as expected", "[group_op][all_of_group]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_local_linear_id();
                CHECK(sycl::all_of_group(it.get_group(), inputs[id], [](int i) { return i <= 4; }));
                CHECK_FALSE(sycl::all_of_group(it.get_group(), inputs[id], [](int i) { return i < 4; }));
                check_group_op_sequence(
                    it.get_group(), {detail::group_operation_id::all_of, detail::group_operation_id::all_of});
            });
        });

        // bool-only signature variant
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_local_linear_id();
                CHECK(sycl::all_of_group(it.get_group(), inputs[id] <= 4));
                CHECK_FALSE(sycl::all_of_group(it.get_group(), inputs[id] < 4));
                check_group_op_sequence(
                    it.get_group(), {detail::group_operation_id::all_of, detail::group_operation_id::all_of});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_sub_group().get_local_linear_id();
                CHECK(sycl::all_of_group(it.get_sub_group(), inputs[id], [](int i) { return i <= 4; }));
                CHECK_FALSE(sycl::all_of_group(it.get_sub_group(), inputs[id], [](int i) { return i < 4; }));
                check_group_op_sequence(
                    it.get_sub_group(), {detail::group_operation_id::all_of, detail::group_operation_id::all_of});
            });
        });
    }
}

TEST_CASE("Group joint_none_of behaves as expected", "[group_op][joint_none_of]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_none_of(it.get_group(), inputs, inputs + 4, [](int i) { return i > 4; }));
                CHECK_FALSE(sycl::joint_none_of(it.get_group(), inputs, inputs + 4, [](int i) { return i == 3; }));
                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::joint_none_of, detail::group_operation_id::joint_none_of});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_none_of(it.get_sub_group(), inputs, inputs + 4, [](int i) { return i > 4; }));
                CHECK_FALSE(sycl::joint_none_of(it.get_sub_group(), inputs, inputs + 4, [](int i) { return i == 3; }));
                check_group_op_sequence(it.get_sub_group(),
                    {detail::group_operation_id::joint_none_of, detail::group_operation_id::joint_none_of});
            });
        });
    }
}

TEST_CASE("Group none_of_group behaves as expected", "[group_op][none_of_group]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_local_linear_id();
                CHECK(sycl::none_of_group(it.get_group(), inputs[id], [](int i) { return i > 4; }));
                CHECK_FALSE(sycl::none_of_group(it.get_group(), inputs[id], [](int i) { return i == 3; }));
                check_group_op_sequence(
                    it.get_group(), {detail::group_operation_id::none_of, detail::group_operation_id::none_of});
            });
        });

        // bool-only signature variant
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_local_linear_id();
                CHECK(sycl::none_of_group(it.get_group(), inputs[id] > 4));
                CHECK_FALSE(sycl::none_of_group(it.get_group(), inputs[id] == 3));
                check_group_op_sequence(
                    it.get_group(), {detail::group_operation_id::none_of, detail::group_operation_id::none_of});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_sub_group().get_local_linear_id();
                CHECK(sycl::none_of_group(it.get_sub_group(), inputs[id], [](int i) { return i > 4; }));
                CHECK_FALSE(sycl::none_of_group(it.get_sub_group(), inputs[id], [](int i) { return i == 3; }));
                check_group_op_sequence(
                    it.get_sub_group(), {detail::group_operation_id::none_of, detail::group_operation_id::none_of});
            });
        });
    }
}

TEST_CASE("Group shift operation behave as expected", "[group_op][shift]") {
    int inputs[4] = {1, 2, 3, 4};
    detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};

    SECTION("Left shift") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_sub_group().get_local_linear_id();
                auto val = sycl::shift_group_left(it.get_sub_group(), inputs[id], 1);
                if(id < 3) {
                    CHECK(val == inputs[id + 1]);
                } else {
                    CHECK(val == detail::unspecified<int>);
                }
                check_group_op_sequence(it.get_sub_group(), {detail::group_operation_id::shift_left});
            });
        });
    }
    SECTION("Right shift") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_sub_group().get_local_linear_id();
                auto val = sycl::shift_group_right(it.get_sub_group(), inputs[id], 2);
                if(id >= 2) {
                    CHECK(val == inputs[id - 2]);
                } else {
                    CHECK(val == detail::unspecified<int>);
                }
                check_group_op_sequence(it.get_sub_group(), {detail::group_operation_id::shift_right});
            });
        });
    }
}

TEST_CASE("Group permute behaves as expected", "[group_op][permute]") {
    int inputs[4] = {1, 2, 3, 4};
    detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};

    sycl::queue().submit([&inputs](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
            auto id = it.get_sub_group().get_local_linear_id();
            auto val = sycl::permute_group(it.get_sub_group(), inputs[id], 0b0101u);
            auto target = id ^ 0b0101u;
            if(target < 4) {
                CHECK(val == inputs[target]);
            } else {
                CHECK(val == detail::unspecified<int>);
            }
            check_group_op_sequence(it.get_sub_group(), {detail::group_operation_id::permute});
        });
    });
}

TEST_CASE("Group select behaves as expected", "[group_op][select]") {
    int inputs[4] = {1, 2, 3, 4};
    detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};

    sycl::queue().submit([&inputs](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
            auto id = it.get_sub_group().get_local_linear_id();
            auto val = sycl::select_from_group(it.get_sub_group(), inputs[id], 3 - id);
            CHECK(val == inputs[3 - id]);
            check_group_op_sequence(it.get_sub_group(), {detail::group_operation_id::select});
        });
    });
}

TEST_CASE("Group joint_reduce behaves as expected", "[group_op][joint_reduce]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_reduce(it.get_group(), inputs, inputs + 4, sycl::plus<int>{}) == 10);
                CHECK(sycl::joint_reduce(it.get_group(), inputs, inputs + 3, sycl::multiplies<int>{}) == 6);
                CHECK(sycl::joint_reduce(it.get_group(), inputs, inputs + 3, 42, sycl::maximum<int>{}) == 42);
                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::joint_reduce, detail::group_operation_id::joint_reduce,
                        detail::group_operation_id::joint_reduce});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                CHECK(sycl::joint_reduce(it.get_sub_group(), inputs, inputs + 4, sycl::plus<int>{}) == 10);
                CHECK(sycl::joint_reduce(it.get_sub_group(), inputs, inputs + 3, sycl::multiplies<int>{}) == 6);
                CHECK(sycl::joint_reduce(it.get_sub_group(), inputs, inputs + 3, 42, sycl::maximum<int>{}) == 42);
                check_group_op_sequence(it.get_sub_group(),
                    {detail::group_operation_id::joint_reduce, detail::group_operation_id::joint_reduce,
                        detail::group_operation_id::joint_reduce});
            });
        });
    }
}

TEST_CASE("Group reduce_over_group behaves as expected", "[group_op][reduce_over_group]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_group().get_local_linear_id();
                CHECK(sycl::reduce_over_group(it.get_group(), inputs[id], sycl::plus<int>{}) == 10);
                CHECK(sycl::reduce_over_group(it.get_group(), inputs[id] - 1, sycl::multiplies<int>{}) == 0);
                CHECK(sycl::reduce_over_group(it.get_group(), inputs[id], -1, sycl::minimum<int>{}) == -1);
                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::reduce, detail::group_operation_id::reduce,
                        detail::group_operation_id::reduce});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                auto id = it.get_sub_group().get_local_linear_id();
                CHECK(sycl::reduce_over_group(it.get_sub_group(), inputs[id], sycl::plus<int>{}) == 10);
                CHECK(sycl::reduce_over_group(it.get_sub_group(), inputs[id] - 1, sycl::multiplies<int>{}) == 0);
                CHECK(sycl::reduce_over_group(it.get_sub_group(), inputs[id], -1, sycl::minimum<int>{}) == -1);
                check_group_op_sequence(it.get_sub_group(),
                    {detail::group_operation_id::reduce, detail::group_operation_id::reduce,
                        detail::group_operation_id::reduce});
            });
        });
    }
}

TEST_CASE("Group joint scans behave as expected", "[group_op][joint_exclusive_scan][joint_inclusive_scan]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                std::vector<int> outputs = {0, 0, 0, 0};
                sycl::joint_exclusive_scan(it.get_group(), inputs, inputs + 4, outputs.data(), sycl::plus<int>{});
                CHECK(outputs == std::vector<int>({0, 1, 3, 6}));
                sycl::joint_inclusive_scan(it.get_group(), inputs, inputs + 4, outputs.data(), sycl::plus<int>{});
                CHECK(outputs == std::vector<int>({1, 3, 6, 10}));
                sycl::joint_exclusive_scan(it.get_group(), inputs, inputs + 4, outputs.data(), sycl::maximum<int>{});
                CHECK(outputs == std::vector<int>({std::numeric_limits<int>::lowest(), 1, 2, 3}));
                sycl::joint_inclusive_scan(it.get_group(), inputs, inputs + 4, outputs.data(), sycl::minimum<int>{});
                CHECK(outputs == std::vector<int>({1, 1, 1, 1}));

                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::joint_exclusive_scan, detail::group_operation_id::joint_inclusive_scan,
                        detail::group_operation_id::joint_exclusive_scan,
                        detail::group_operation_id::joint_inclusive_scan});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                std::vector<int> outputs = {0, 0, 0, 0};
                auto sg = it.get_sub_group();
                sycl::joint_exclusive_scan(sg, inputs, inputs + 4, outputs.data(), sycl::plus<int>{});
                CHECK(outputs == std::vector<int>({0, 1, 3, 6}));
                sycl::joint_inclusive_scan(sg, inputs, inputs + 4, outputs.data(), sycl::plus<int>{});
                CHECK(outputs == std::vector<int>({1, 3, 6, 10}));
                sycl::joint_exclusive_scan(sg, inputs, inputs + 4, outputs.data(), sycl::maximum<int>{});
                CHECK(outputs == std::vector<int>({std::numeric_limits<int>::lowest(), 1, 2, 3}));
                sycl::joint_inclusive_scan(sg, inputs, inputs + 4, outputs.data(), sycl::minimum<int>{});
                CHECK(outputs == std::vector<int>({1, 1, 1, 1}));

                check_group_op_sequence(sg,
                    {detail::group_operation_id::joint_exclusive_scan, detail::group_operation_id::joint_inclusive_scan,
                        detail::group_operation_id::joint_exclusive_scan,
                        detail::group_operation_id::joint_inclusive_scan});
            });
        });
    }

    SECTION("With init value") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                std::vector<int> outputs = {0, 0, 0, 0};
                sycl::joint_exclusive_scan(it.get_group(), inputs, inputs + 4, outputs.data(), 5, sycl::plus<int>{});
                CHECK(outputs == std::vector<int>({5, 6, 8, 11}));
                sycl::joint_inclusive_scan(it.get_group(), inputs, inputs + 4, outputs.data(), 5, sycl::plus<int>{});
                CHECK(outputs == std::vector<int>({6, 8, 11, 15}));
                sycl::joint_exclusive_scan(it.get_group(), inputs, inputs + 4, outputs.data(), 2, sycl::maximum<int>{});
                CHECK(outputs == std::vector<int>({2, 2, 2, 3}));
                sycl::joint_inclusive_scan(
                    it.get_group(), inputs, inputs + 4, outputs.data(), -1, sycl::minimum<int>{});
                CHECK(outputs == std::vector<int>({-1, -1, -1, -1}));

                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::joint_exclusive_scan, detail::group_operation_id::joint_inclusive_scan,
                        detail::group_operation_id::joint_exclusive_scan,
                        detail::group_operation_id::joint_inclusive_scan});
            });
        });
    }
}


TEST_CASE("Group scans behave as expected", "[group_op][exclusive_scan_over_group][inclusive_scan_over_group]") {
    int inputs[4] = {1, 2, 3, 4};

    SECTION("For work groups") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                const auto id = it.get_group().get_local_linear_id();
                CHECK(sycl::exclusive_scan_over_group(it.get_group(), inputs[id], sycl::plus<int>{})
                    == std::vector<int>({0, 1, 3, 6})[id]);
                CHECK(sycl::inclusive_scan_over_group(it.get_group(), inputs[id], sycl::plus<int>{})
                    == std::vector<int>({1, 3, 6, 10})[id]);
                CHECK(sycl::exclusive_scan_over_group(it.get_group(), inputs[id], sycl::maximum<int>{})
                    == std::vector<int>({std::numeric_limits<int>::lowest(), 1, 2, 3})[id]);
                CHECK(sycl::inclusive_scan_over_group(it.get_group(), inputs[id], sycl::minimum<int>{})
                    == std::vector<int>({1, 1, 1, 1})[id]);

                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::exclusive_scan, detail::group_operation_id::inclusive_scan,
                        detail::group_operation_id::exclusive_scan, detail::group_operation_id::inclusive_scan});
            });
        });
    }

    SECTION("For subgroups") {
        detail::configure_temporarily cfg{detail::config::max_sub_group_size, 4u};
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 8}, [&inputs](sycl::nd_item<1> it) {
                const auto id = it.get_sub_group().get_local_linear_id();
                CHECK(sycl::exclusive_scan_over_group(it.get_sub_group(), inputs[id], sycl::plus<int>{})
                    == std::vector<int>({0, 1, 3, 6})[id]);
                CHECK(sycl::inclusive_scan_over_group(it.get_sub_group(), inputs[id], sycl::plus<int>{})
                    == std::vector<int>({1, 3, 6, 10})[id]);
                CHECK(sycl::exclusive_scan_over_group(it.get_sub_group(), inputs[id], sycl::maximum<int>{})
                    == std::vector<int>({std::numeric_limits<int>::lowest(), 1, 2, 3})[id]);
                CHECK(sycl::inclusive_scan_over_group(it.get_sub_group(), inputs[id], sycl::minimum<int>{})
                    == std::vector<int>({1, 1, 1, 1})[id]);

                check_group_op_sequence(it.get_sub_group(),
                    {detail::group_operation_id::exclusive_scan, detail::group_operation_id::inclusive_scan,
                        detail::group_operation_id::exclusive_scan, detail::group_operation_id::inclusive_scan});
            });
        });
    }

    SECTION("With init value") {
        sycl::queue().submit([&inputs](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>{8, 4}, [&inputs](sycl::nd_item<1> it) {
                const auto id = it.get_group().get_local_linear_id();
                CHECK(sycl::exclusive_scan_over_group(it.get_group(), inputs[id], 5, sycl::plus<int>{})
                    == std::vector<int>({5, 6, 8, 11})[id]);
                CHECK(sycl::inclusive_scan_over_group(it.get_group(), inputs[id], 5, sycl::plus<int>{})
                    == std::vector<int>({6, 8, 11, 15})[id]);
                CHECK(sycl::exclusive_scan_over_group(it.get_group(), inputs[id], 2, sycl::maximum<int>{})
                    == std::vector<int>({2, 2, 2, 3})[id]);
                CHECK(sycl::inclusive_scan_over_group(it.get_group(), inputs[id], -1, sycl::minimum<int>{})
                    == std::vector<int>({-1, -1, -1, -1})[id]);

                check_group_op_sequence(it.get_group(),
                    {detail::group_operation_id::exclusive_scan, detail::group_operation_id::inclusive_scan,
                        detail::group_operation_id::exclusive_scan, detail::group_operation_id::inclusive_scan});
            });
        });
    }
}
