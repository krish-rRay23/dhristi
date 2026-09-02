add_library(drishti_analysis STATIC
    analysis.cpp
    mlir_analysis.cpp
    gpu_lowering.cpp
    llvm_integration.cpp
)

target_link_libraries(drishti_analysis PUBLIC drishti_common drishti_core drishti_backends)

if(DRISHTI_HAVE_MLIR)
    target_include_directories(drishti_analysis SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
    target_compile_definitions(drishti_analysis PUBLIC DRISHTI_HAS_MLIR=1)
    target_compile_options(drishti_analysis PUBLIC -fno-rtti)
    if(LLVM_DEFINITIONS)
        separate_arguments(LLVM_DEFS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
        target_compile_definitions(drishti_analysis PUBLIC ${LLVM_DEFS_LIST})
    endif()
    target_link_libraries(drishti_analysis PUBLIC
        ${DRISHTI_MLIR_LIBS} ${DRISHTI_LLVM_LIBS})
endif()

add_library(drishti::analysis ALIAS drishti_analysis)
