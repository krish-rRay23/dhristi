#ifndef DRISHTI_ANALYSIS_GPU_LOWERING_H
#define DRISHTI_ANALYSIS_GPU_LOWERING_H

// Phase 9: MLIR -> PTX lowering for true compiler experiments.
//
// Takes MLIR source plus a pass-pipeline string and produces PTX text ready
// for the existing CUDA driver harness:
//
//   MLIR -> pipeline passes (incl. affine/GPU lowering) -> gpu.module(s)
//        -> per-kernel launch specs from host launch ops -> NVVM translation
//        -> LLVM IR -> NVPTX backend -> PTX
//
// No CUDA toolkit is required: PTX comes from LLVM's NVPTX target and the
// driver JIT-compiles it at load time.
//
// Launch specs are read from the host-side gpu.launch_func ops (constant
// index operands + func-arg positions of memref operands), so callers never
// hardcode compiler-generated kernel names, signatures, or argument orders.
// Pure host-side step: needs no GPU.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace drishti::analysis {

// One kernel launch as the compiler described it on the host side.
struct KernelLaunchSpec {
    std::string entry;                 // unique kernel name (renamed at lowering)
    std::vector<std::int64_t> index_args;  // leading i64 params, in order
    std::vector<int> buffer_slots;         // func-arg positions of memref params
};

struct LoweredKernels {
    bool ok = false;
    std::string error;
    std::string pipeline_used;
    std::string ptx;                      // concatenated PTX for all gpu.modules
    std::vector<std::string> entry_names;  // in PTX emission order
    std::vector<KernelLaunchSpec> launches;  // host launch order
    [[nodiscard]] std::size_t num_kernels() const noexcept { return entry_names.size(); }
};

// Lower MLIR source through the given pipeline to PTX for sm_<major><minor>.
// Kernel funcs are renamed to unique names; launch specs are extracted from
// the host launch ops. Returns ok=false with error on any failure.
LoweredKernels lower_to_ptx(std::string_view source, std::string_view pipeline,
                            int cc_major, int cc_minor, std::string* err);

// Reference workload for the fusion experiment: d = (a+b)*c as two
// perfectly nested affine.for nests over memref<DxDxf32> (func @fusedemo),
// where D*D == num_elements (num_elements must be a perfect square).
[[nodiscard]] std::string fusion_workload_mlir(std::size_t num_elements);

// Baseline pipeline: cleanup + affine->GPU lowering, no fusion.
[[nodiscard]] std::string fusion_baseline_pipeline();

// Transformed pipeline: exactly one added transformation,
// affine-loop-fusion, before the same GPU lowering.
[[nodiscard]] std::string fusion_transformed_pipeline();

// Registers exactly the extra passes used by the GPU pipelines
// (affine-loop-fusion, convert-affine-for-to-gpu, gpu-kernel-outlining,
// convert-gpu-to-nvvm) on top of registerTransformsPasses(). Idempotent
// alongside repeated calls; shared by the provenance engine and lowering.
void register_gpu_pipeline_passes();

}  // namespace drishti::analysis

#endif
