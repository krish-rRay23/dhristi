add_executable(drishti
    main.cpp
)

target_link_libraries(drishti PRIVATE
    drishti::core
    drishti::analysis
    drishti::provenance
    drishti::profiling
    drishti::diagnosis
    drishti::optimizer
    drishti::backends
    drishti::correlation
    drishti::benchmark
    drishti::triton
)

set_target_properties(drishti PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
