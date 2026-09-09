add_executable(test_suggest
    test_suggest.cpp
)

target_link_libraries(test_suggest PUBLIC
    drishti_optimizer
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_suggest COMMAND test_suggest)
set_tests_properties(test_suggest PROPERTIES RUN_SERIAL TRUE)

add_executable(test_experiments
    test_experiments.cpp
)

target_link_libraries(test_experiments PUBLIC
    drishti_optimizer
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_experiments COMMAND test_experiments)
set_tests_properties(test_experiments PROPERTIES RUN_SERIAL TRUE)

add_executable(test_compiler_experiment
    test_compiler_experiment.cpp
)

target_link_libraries(test_compiler_experiment PUBLIC
    drishti_optimizer
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_compiler_experiment COMMAND test_compiler_experiment)
set_tests_properties(test_compiler_experiment PROPERTIES RUN_SERIAL TRUE)

add_executable(test_cost_model
    test_cost_model.cpp
)

target_link_libraries(test_cost_model PUBLIC
    drishti_optimizer
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_cost_model COMMAND test_cost_model)

add_executable(test_search
    test_search.cpp
)

target_link_libraries(test_search PUBLIC
    drishti_optimizer
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_search COMMAND test_search)

add_executable(test_calibration
    test_calibration.cpp
)

target_link_libraries(test_calibration PUBLIC
    drishti_optimizer
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_calibration COMMAND test_calibration)
