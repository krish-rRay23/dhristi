add_library(drishti_backends STATIC
    backends.cpp
    cuda/cuda_backend.cpp
    rocm/rocm_backend.cpp
)

target_link_libraries(drishti_backends PUBLIC drishti_common drishti_core drishti_profiling)

if(DRISHTI_HAVE_LLVM)
    target_include_directories(drishti_backends SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    if(LLVM_DEFINITIONS)
        separate_arguments(LLVM_DEFS_BK NATIVE_COMMAND ${LLVM_DEFINITIONS})
        target_compile_definitions(drishti_backends PUBLIC ${LLVM_DEFS_BK})
    endif()
    target_compile_definitions(drishti_backends PRIVATE DRISHTI_HAS_LLVM=1)
endif()

add_library(drishti::backends ALIAS drishti_backends)
