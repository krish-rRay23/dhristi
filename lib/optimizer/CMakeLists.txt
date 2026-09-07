add_library(drishti_optimizer STATIC
    optimizer.cpp
    suggest.cpp
    experiment.cpp
    compiler_experiment.cpp
    cost_model.cpp
    search.cpp
)

target_link_libraries(drishti_optimizer PUBLIC drishti_common drishti_core drishti_diagnosis)

if(DRISHTI_HAVE_MLIR)
    target_include_directories(drishti_optimizer SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
    if(LLVM_DEFINITIONS)
        separate_arguments(LLVM_DEFS_OPT NATIVE_COMMAND ${LLVM_DEFINITIONS})
        target_compile_definitions(drishti_optimizer PUBLIC ${LLVM_DEFS_OPT})
    endif()
    target_compile_definitions(drishti_optimizer PRIVATE DRISHTI_HAS_MLIR=1)
    target_link_libraries(drishti_optimizer PUBLIC MLIRSupport)
endif()

add_library(drishti::optimizer ALIAS drishti_optimizer)
