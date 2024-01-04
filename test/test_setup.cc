#include <simsycl/schedule.hh>
#include <simsycl/system.hh>

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>


struct global_setup_and_teardown : Catch::EventListenerBase {
    using EventListenerBase::EventListenerBase;

    void testCasePartialStarting(const Catch::TestCaseInfo & /* test_info */, uint64_t /* part_number */) override {
        simsycl::configure_system(simsycl::builtin_system);
        simsycl::set_cooperative_schedule(std::make_unique<simsycl::round_robin_schedule>());
    }
};

CATCH_REGISTER_LISTENER(global_setup_and_teardown);
