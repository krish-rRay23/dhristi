add_library(drishti_triton STATIC
    triton_integration.cpp
)

target_compile_definitions(drishti_triton PRIVATE
    DRISHTI_TRITON_COMPILER_SCRIPT_PATH="${DRISHTI_ROOT_DIR}/tools/triton/drishti_triton_compiler.py"
)

target_link_libraries(drishti_triton PUBLIC
    drishti_common
    drishti_core
    drishti_backends
    drishti_provenance
    drishti_profiling
    drishti_diagnosis
    drishti_optimizer
    drishti_correlation
)

add_library(drishti::triton ALIAS drishti_triton)
