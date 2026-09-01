add_library(drishti_core STATIC
    version.cpp
    export.cpp
)

target_link_libraries(drishti_core PUBLIC drishti_common)

add_library(drishti::core ALIAS drishti_core)
