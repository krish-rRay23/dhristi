add_library(drishti_diagnosis STATIC
    diagnosis.cpp
    root_cause.cpp
    cross_vendor.cpp
)

target_link_libraries(drishti_diagnosis PUBLIC drishti_common drishti_core drishti_correlation drishti_optimizer)

add_library(drishti::diagnosis ALIAS drishti_diagnosis)
