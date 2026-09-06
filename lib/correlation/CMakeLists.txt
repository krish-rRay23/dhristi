add_library(drishti_correlation STATIC
    correlation.cpp
)

target_link_libraries(drishti_correlation PUBLIC
    drishti_common
    drishti_core
    drishti_analysis
    drishti_profiling
    drishti_backends
    drishti_provenance
)

add_library(drishti::correlation ALIAS drishti_correlation)
