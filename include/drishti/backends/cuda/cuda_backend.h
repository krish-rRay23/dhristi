#ifndef DRISHTI_BACKENDS_CUDA_CUDA_BACKEND_H
#define DRISHTI_BACKENDS_CUDA_CUDA_BACKEND_H

// NVIDIA CUDA backend (Phase 4). All NVIDIA-specific code lives behind this
// interface; no CUDA headers are required at build time. At runtime the
// backend dynamically loads the CUDA driver library (nvcuda.dll on Windows,
// libcuda.so.1 on Linux) and JIT-compiles an embedded PTX kernel, so it
// works with only the GPU driver installed (no CUDA toolkit needed).
//
// CUPTI is probed opportunistically: when its shared library is present the
// backend reports its version, otherwise it reports unavailable and falls
// back to CUDA-event timing. No AMD/ROCm code lives here.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "drishti/profiling/gpu_metrics.h"

namespace drishti::backends::cuda {

// Lightweight presence check used by BackendRegistry. Loads the driver,
// runs cuInit and requires device count > 0. Never throws.
[[nodiscard]] bool device_present() noexcept;

// Full device query. Returns true and fills `out` on success; on failure
// returns false with `out.present == false` and `out.error` set.
bool query_device(profiling::GpuDeviceModel& out);

// CUPTI probe. Never fails: sets `available` and a human-readable `version`
// ("<major>.<minor>" or "unavailable (<reason>)").
void cupti_status(bool& available, std::string& version);

struct VecaddConfig {
    std::size_t num_elements = 1u << 20;  // 1M floats
    int block_size = 256;
    int repeats = 10;
};

// Compile (JIT) and run a representative single-precision vector-add kernel
// (c[i] = a[i] + b[i]) on device 0, timed with CUDA events plus host-side
// timers for transfers. Fills `out` (ok == true on success, incl. result
// verification) or returns false with `err` set. Never throws.
bool run_vecadd_profile(const VecaddConfig& cfg, profiling::GpuProfileMetrics& out,
                        std::string* err);

struct VecaddVariantConfig {
    std::size_t num_elements = 1u << 16;
    int block_size = 256;
    int repeats = 5;
    // Naive app pattern: copy inputs to the device and read the result back
    // on *every* round. false = persistent buffers (copy once, as in
    // run_vecadd_profile). Used for controlled reuse experiments.
    bool recopy_per_launch = false;
};

// Same metrics shape as run_vecadd_profile, so variants compare directly.
// bytes_moved counts total traffic (scales with repeats when recopying).
// Never throws.
bool run_vecadd_variant(const VecaddVariantConfig& cfg,
                        profiling::GpuProfileMetrics& out, std::string* err);

// Phase 9: run compiler-generated PTX (e.g. from MLIR GPU lowering) with the
// same harness discipline. Launches execute in order per repeat; each launch
// carries its own index args and buffer slots as described by the compiler's
// host launch ops. Kernel timings are per launch; metrics cover the whole
// sequence. Buffers 0..num_buffers-2 are inputs (filled from input_fills),
// buffer output_slot is verified against `expected`. Requires exact launch
// geometry matching the compiled mapping. Same metrics shape; never throws.
struct PtxKernelLaunch {
    std::string entry;
    std::vector<std::int64_t> index_args;  // leading i64 params, in order
    std::vector<int> buffer_slots;         // device-buffer indices, in order
};

struct PtxLaunchConfig {
    std::string ptx_text;
    std::vector<PtxKernelLaunch> launches;  // execution order per repeat
    std::string kernel_label = "fusedemo";
    std::size_t num_elements = 65536;
    int grid_size = 256;  // explicit: must match the compiled mapping
    int block_size = 256;
    int repeats = 5;
    int num_buffers = 5;
    int output_slot = 4;
    std::vector<float> input_fills = {1.0f, 2.0f, 0.0f, 3.0f, 0.0f};
    float expected = 9.0f;
    double bytes_per_element = 16.0;
};

struct PerKernelTime {
    std::string entry;
    double ms_avg = 0.0;
    double ms_min = 0.0;
};

struct PtxRunResult {
    profiling::GpuProfileMetrics metrics;
    std::vector<PerKernelTime> per_kernel;  // one per launch, in order
};

bool run_ptx_kernel(const PtxLaunchConfig& cfg, PtxRunResult& out,
                    std::string* err);

// Phase 18: Run Triton-generated PTX kernel directly on the GPU.
struct TritonLaunchConfig {
    std::string ptx_text;
    std::string entry_name;
    std::string workload_name = "fused_add_relu";
    std::size_t num_elements = 65536;
    int grid_size = 256;
    int block_size = 128;  // num_warps * 32
    int num_warps = 4;
    std::size_t shared_mem_bytes = 0;
    int repeats = 5;
    double bytes_per_element = 12.0;  // 2 ld + 1 st = 12 bytes/elem for f32
    double flops_per_element = 2.0;   // 1 add + 1 relu (max) = 2 flops/elem
};

struct TritonRunResult {
    profiling::GpuProfileMetrics metrics;
    bool verified = false;
    double measured_kernel_ms = 0.0;
    std::string error;
};

bool run_triton_kernel(const TritonLaunchConfig& cfg, TritonRunResult& out,
                       std::string* err);

}  // namespace drishti::backends::cuda

#endif
