add_library(drishti_provenance STATIC
    provenance.cpp
)

target_link_libraries(drishti_provenance PUBLIC drishti_common drishti_core)

add_library(drishti::provenance ALIAS drishti_provenance)
