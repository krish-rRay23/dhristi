add_library(drishti_profiling STATIC
    profiling.cpp
    gpu_metrics.cpp
)

target_link_libraries(drishti_profiling PUBLIC drishti_common drishti_core drishti_provenance)

add_library(drishti::profiling ALIAS drishti_profiling)
