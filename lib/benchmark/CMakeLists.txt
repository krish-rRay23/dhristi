add_library(drishti_benchmark STATIC
    benchmark.cpp
    journal_experiments.cpp
)

target_link_libraries(drishti_benchmark PUBLIC drishti_common drishti_core drishti_profiling drishti_diagnosis drishti_optimizer drishti_backends)

if(DRISHTI_HAVE_MLIR)
    target_include_directories(drishti_benchmark SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
    if(LLVM_DEFINITIONS)
        separate_arguments(LLVM_DEFS_OPT NATIVE_COMMAND ${LLVM_DEFINITIONS})
        target_compile_definitions(drishti_benchmark PUBLIC ${LLVM_DEFS_OPT})
    endif()
    target_compile_definitions(drishti_benchmark PRIVATE DRISHTI_HAS_MLIR=1)
    target_link_libraries(drishti_benchmark PUBLIC MLIRSupport)
endif()

add_library(drishti::benchmark ALIAS drishti_benchmark)
