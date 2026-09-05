#ifndef DRISHTI_PROFILING_GPU_METRICS_H
#define DRISHTI_PROFILING_GPU_METRICS_H

// Vendor-neutral GPU profiling metric model (Phase 4).
//
// This is the concise machine-readable/internal model that later diagnosis
// and provenance phases consume. Vendor backends (e.g. backends/cuda) fill
// it in; they must not leak vendor headers through this interface.

#include <cstddef>
#include <string>

#include "drishti/provenance/provenance.h"

namespace drishti::profiling {

struct GpuDeviceModel {
    std::string backend = "unknown";  // e.g. "cuda"
    std::string name;                 // e.g. "NVIDIA GeForce RTX 3050 Laptop GPU"
    bool present = false;
    int compute_major = 0;
    int compute_minor = 0;
    int sm_count = 0;
    int max_threads_per_block = 0;
    long clock_mhz = 0;
    long mem_clock_mhz = 0;
    int mem_bus_width_bits = 0;
    double mem_theoretical_gbps = 0.0;
    unsigned long long total_mem_bytes = 0;
    std::string driver_version;  // e.g. "13.1" (driver API) or driver string
    std::string error;           // set when present == false
};

struct GpuProfileMetrics {
    bool ok = false;
    std::string error;  // set when ok == false

    GpuDeviceModel device;
    std::string kernel;  // e.g. "vecadd"
    std::size_t num_elements = 0;
    int block_size = 0;
    int grid_size = 0;
    int repeats = 0;

    double kernel_ms_avg = 0.0;  // CUDA-event timed, averaged over repeats
    double kernel_ms_min = 0.0;  // fastest single repeat
    double h2d_ms = 0.0;         // host->device copy (host timed)
    double d2h_ms = 0.0;         // device->host copy (host timed)
    double wall_ms = 0.0;        // end-to-end host time incl. alloc/copy/launch
    double gbps_effective = 0.0;  // (bytes read + written) / kernel time
    unsigned long long bytes_moved = 0;

    bool correct = false;              // host-side result verification
    std::string timing_source;         // e.g. "cuda-events"
    bool cupti_available = false;      // true only when CUPTI lib loads
    std::string cupti_version;         // e.g. "12.4" or "unavailable (no CUDA toolkit)"
    std::string notes;                 // backend caveats (WDDM, no CUPTI, ...)
};

// Machine-readable rendering consumed by later phases (diagnosis/provenance).
[[nodiscard]] std::string gpu_metrics_to_json(const GpuProfileMetrics& m);

// Human-readable one-page report for `drishti profile`.
[[nodiscard]] std::string format_gpu_report(const GpuProfileMetrics& m);

// Provenance wiring: represent a GPU profile run as a provenance pass so the
// existing analysis/provenance model can reference it. No diagnosis or
// optimization is performed here (later phases).
[[nodiscard]] drishti::provenance::PassInfo gpu_metrics_to_pass_info(
    const GpuProfileMetrics& m);

}  // namespace drishti::profiling

#endif
