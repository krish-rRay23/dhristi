#ifndef DRISHTI_BACKENDS_ROCM_ROCM_BACKEND_H
#define DRISHTI_BACKENDS_ROCM_ROCM_BACKEND_H

// AMD ROCm / HIP backend (Phase 11).
//
// All AMD-specific code lives behind this interface; no ROCm/HIP SDK headers
// are required at build time. At runtime the backend dynamically loads the
// HIP runtime library (amdhip64.dll on Windows, libamdhip64.so on Linux).
//
// If ROCm hardware/driver is absent locally, a mock validation mode is provided
// to test interface compliance and normalized metric emission cleanly.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "drishti/profiling/gpu_metrics.h"

namespace drishti::backends::rocm {

// Presence check for ROCm/HIP runtime/driver.
[[nodiscard]] bool device_present() noexcept;

// Enable/disable mock validation mode when physical ROCm GPU is absent.
void set_mock_mode(bool enable) noexcept;
[[nodiscard]] bool is_mock_mode() noexcept;

// Full ROCm device query.
bool query_device(profiling::GpuDeviceModel& out);

// ROCm runtime status probe.
void rocm_status(bool& available, std::string& version);

struct VecaddConfig {
    std::size_t num_elements = 1u << 20;  // 1M floats
    int block_size = 256;
    int repeats = 10;
};

// Run representative single-precision vector-add kernel on ROCm/HIP.
bool run_vecadd_profile(const VecaddConfig& cfg, profiling::GpuProfileMetrics& out,
                        std::string* err);

struct VecaddVariantConfig {
    std::size_t num_elements = 1u << 16;
    int block_size = 256;
    int repeats = 5;
    bool recopy_per_launch = false;
};

bool run_vecadd_variant(const VecaddVariantConfig& cfg,
                        profiling::GpuProfileMetrics& out, std::string* err);

struct HipKernelLaunch {
    std::string entry;
    std::vector<std::int64_t> index_args;
    std::vector<int> buffer_slots;
};

struct HipLaunchConfig {
    std::string hsaco_or_code_text;
    std::vector<HipKernelLaunch> launches;
    std::string kernel_label = "fusedemo";
    std::size_t num_elements = 65536;
    int grid_size = 256;
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

struct HipRunResult {
    profiling::GpuProfileMetrics metrics;
    std::vector<PerKernelTime> per_kernel;
};

bool run_hip_kernel(const HipLaunchConfig& cfg, HipRunResult& out,
                    std::string* err);

}  // namespace drishti::backends::rocm

#endif
