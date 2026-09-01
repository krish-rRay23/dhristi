cmake_minimum_required(VERSION 3.24)

project(Drishti
    VERSION 0.1.0
    DESCRIPTION "Dṛṣṭi - ML Compiler and GPU Performance Analysis Framework"
    HOMEPAGE_URL "https://github.com/drishti-ml/drishti"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "[Drishti] Configuring ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "[Drishti] Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "[Drishti] C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CTest)

option(DRISHTI_BUILD_TESTS "Build unit tests (requires GoogleTest)" ON)
option(DRISHTI_BUILD_CLI "Build the drishti command-line tool" ON)
option(DRISHTI_ENABLE_LLVM "Enable LLVM/MLIR backend if found" ON)
option(DRISHTI_ENABLE_CUDA "Enable NVIDIA CUDA backend (driver-loaded, no toolkit required)" ON)
option(DRISHTI_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(DRISHTI_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    add_compile_options(/W4 /permissive- /EHsc /utf-8)
    if(DRISHTI_WARNINGS_AS_ERRORS)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -ffunction-sections -fdata-sections)
    add_link_options(-Wl,--gc-sections -Wl,--allow-multiple-definition)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--allow-multiple-definition")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
    set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
    if(DRISHTI_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()


set(DRISHTI_HAVE_LLVM OFF)
set(DRISHTI_HAVE_MLIR OFF)

if(DRISHTI_ENABLE_LLVM)
    find_package(LLVM CONFIG QUIET)
    if(LLVM_FOUND)
        message(STATUS "[Drishti] LLVM found: ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
        set(DRISHTI_HAVE_LLVM ON)
        list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
        include(AddLLVM)
    else()
        message(STATUS "[Drishti] LLVM not found - LLVM/MLIR backends will be disabled")
    endif()

    if(DRISHTI_HAVE_LLVM)
        find_package(MLIR CONFIG QUIET)
        if(MLIR_FOUND)
            message(STATUS "[Drishti] MLIR found: ${MLIR_VERSION} (${MLIR_DIR})")
            set(DRISHTI_HAVE_MLIR ON)
            list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
            foreach(_tgt MLIRGPUTransforms MLIRGPUToNVVMTransforms MLIRGPUDialect MLIRGPUUtils MLIRSCFToGPU MLIRNVVMTarget MLIRTargetLLVM MLIRTargetLLVMIRExport)
                if(TARGET ${_tgt})
                    get_target_property(_libs ${_tgt} INTERFACE_LINK_LIBRARIES)
                    if(_libs)
                        list(FILTER _libs EXCLUDE REGEX "ROCDL|AMDGPU|SPIRV|XeGPU|XeVM")
                        set_target_properties(${_tgt} PROPERTIES INTERFACE_LINK_LIBRARIES "${_libs}")
                    endif()
                endif()
            endforeach()
            set(DRISHTI_MLIR_LIBS
                MLIRGPUTransforms
                MLIRGPUToNVVMTransforms
                MLIRLLVMCommonConversion
                MLIRArithToLLVM
                MLIRIndexToLLVM
                MLIRAffineToStandard
                MLIRReconcileUnrealizedCasts
                MLIRGPUToGPURuntimeTransforms
                MLIRSCFToGPU
                MLIRGPUUtils
                MLIRNVVMToLLVMIRTranslation
                MLIRGPUToLLVMIRTranslation
                MLIRLLVMToLLVMIRTranslation
                MLIRTargetLLVMIRExport
                MLIRBuiltinToLLVMIRTranslation
                MLIRToLLVMIRTranslationRegistration
                MLIRLLVMDialect
                MLIRNVVMDialect
                MLIRNVGPUDialect
                MLIRNVGPUTransforms
                MLIRNVGPUUtils
                MLIRGPUDialect
                MLIRAffineTransforms
                MLIRAffineAnalysis
                MLIRAffineUtils
                MLIRAffineDialect
                MLIRTosaDialect
                MLIRVectorDialect
                MLIRTensorDialect
                MLIRLinalgDialect
                MLIRMathDialect
                MLIRControlFlowDialect
                MLIRAsyncDialect
                MLIRUBDialect
                MLIRComplexDialect
                MLIRMemRefTransforms
                MLIRMemRefDialect
                MLIRArithTransforms
                MLIRArithDialect
                MLIRFuncTransforms
                MLIRFuncDialect
                MLIRSCFDialect
                MLIRSCFUtils
                MLIRTransforms
                MLIRTransformUtils
                MLIRAnalysis
                MLIRPass
                MLIRParser
                MLIRIR
                MLIRSupport
            )
        else()
            message(STATUS "[Drishti] MLIR not found - MLIR passes will be disabled")
        endif()
        # Phase 17 & Phase 9: LLVM & NVPTX components for MLIR -> LLVM IR -> Passes -> PTX
        if(TARGET LLVM)
            set(DRISHTI_LLVM_LIBS LLVM)
        elseif(EXISTS "C:/msys64/ucrt64/lib/libLLVM-22.dll.a")
            set(DRISHTI_LLVM_LIBS "C:/msys64/ucrt64/lib/libLLVM-22.dll.a")
        else()
            llvm_map_components_to_libnames(DRISHTI_LLVM_LIBS
                Passes
                Core
                Support
                Analysis
                TransformUtils
                ScalarOpts
                InstCombine
                Target
                TargetParser
                MC
                NVPTX
            )
        endif()
        message(STATUS "[Drishti] LLVM libs: ${DRISHTI_LLVM_LIBS}")
    endif()
else()
    message(STATUS "[Drishti] LLVM/MLIR disabled by DRISHTI_ENABLE_LLVM=OFF")
endif()

set(DRISHTI_HAVE_LLVM_INT 0)
set(DRISHTI_HAVE_MLIR_INT 0)
set(DRISHTI_HAVE_CUDA_INT 0)
if(DRISHTI_HAVE_LLVM)
    set(DRISHTI_HAVE_LLVM_INT 1)
endif()
if(DRISHTI_HAVE_MLIR)
    set(DRISHTI_HAVE_MLIR_INT 1)
endif()
# The CUDA backend needs no toolkit at build time (driver dlopen + PTX JIT),
# so compiled support is unconditional unless explicitly disabled.
if(DRISHTI_ENABLE_CUDA)
    set(DRISHTI_HAVE_CUDA_INT 1)
    message(STATUS "[Drishti] CUDA backend: enabled (runtime driver load, toolkit optional)")
else()
    message(STATUS "[Drishti] CUDA backend: disabled by DRISHTI_ENABLE_CUDA=OFF")
endif()

configure_file(
    ${DRISHTI_ROOT_DIR}/cmake/drishti_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/drishti/core/config.h
    @ONLY
)

add_library(drishti_common INTERFACE)
target_include_directories(drishti_common INTERFACE
    $<BUILD_INTERFACE:${DRISHTI_ROOT_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(drishti_common INTERFACE cxx_std_20)


add_subdirectory(lib/core)
add_subdirectory(lib/analysis)
add_subdirectory(lib/provenance)
add_subdirectory(lib/profiling)
add_subdirectory(lib/diagnosis)
add_subdirectory(lib/optimizer)
add_subdirectory(lib/backends)
add_subdirectory(lib/correlation)
add_subdirectory(lib/benchmark)
add_subdirectory(lib/triton)

if(DRISHTI_BUILD_CLI)
    add_subdirectory(tools/drishti)
endif()

if(DRISHTI_HAVE_MLIR)
    add_subdirectory(tools/proto)
endif()

if(DRISHTI_BUILD_TESTS AND BUILD_TESTING)
    enable_testing()
    message(STATUS "[Drishti] Fetching GoogleTest...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    add_subdirectory(tests/core)
    if(DRISHTI_HAVE_MLIR)
        add_subdirectory(tests/analysis)
    endif()
    add_subdirectory(tests/provenance)
    add_subdirectory(tests/profiling)
    add_subdirectory(tests/correlation)
    add_subdirectory(tests/optimizer)
    add_subdirectory(tests/diagnosis)
    add_subdirectory(tests/backends)
    add_subdirectory(tests/benchmark)
    add_subdirectory(tests/triton)
endif()

message(STATUS "")
message(STATUS "=============================================")
message(STATUS "  Dṛṣṭi Configuration Summary")
message(STATUS "=============================================")
message(STATUS "  Version          : ${PROJECT_VERSION}")
message(STATUS "  C++ Standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build CLI        : ${DRISHTI_BUILD_CLI}")
message(STATUS "  Build Tests      : ${DRISHTI_BUILD_TESTS}")
message(STATUS "  LLVM Found       : ${DRISHTI_HAVE_LLVM}")
message(STATUS "  MLIR Found       : ${DRISHTI_HAVE_MLIR}")
message(STATUS "  CUDA Backend     : ${DRISHTI_HAVE_CUDA_INT}")
message(STATUS "=============================================")
message(STATUS "")
