#ifndef DRISHTI_DIAGNOSIS_CROSS_VENDOR_H
#define DRISHTI_DIAGNOSIS_CROSS_VENDOR_H

// Phase 12: Cross-Vendor Performance Intelligence for Drishti.
//
// Analyzes and correlates normalized performance metrics from NVIDIA CUDA
// and AMD ROCm backends. Identifies performance differences, memory/compute/launch
// bottlenecks, and separates portable architectural findings from vendor-specific
// hardware/driver traits.

#include <string>
#include <vector>

#include "drishti/diagnosis/root_cause.h"
#include "drishti/profiling/gpu_metrics.h"

namespace drishti::diagnosis {

struct CrossVendorFinding {
    std::string title;
    std::string category;     // "performance", "memory", "compute", "launch", "driver"
    bool is_portable = true;  // true if finding is architectural; false if vendor-specific
    std::string nv_trait;     // NVIDIA value / trait description
    std::string amd_trait;    // AMD value / trait description
    std::string explanation;
};

struct CrossVendorReport {
    bool ok = false;
    std::string error;
    std::string workload;

    // Devices
    profiling::GpuDeviceModel nv_device;
    profiling::GpuDeviceModel amd_device;
    bool nv_is_mock = false;
    bool amd_is_mock = false;

    // Measured Metrics
    profiling::GpuProfileMetrics nv_metrics;
    profiling::GpuProfileMetrics amd_metrics;

    // Computed Comparisons
    double latency_ratio_nv_vs_amd = 0.0;     // nv_ms / amd_ms (>1 means AMD faster)
    double speedup_percent_amd_vs_nv = 0.0;   // (nv_ms - amd_ms) / nv_ms * 100
    double bandwidth_nv_gbps = 0.0;
    double bandwidth_amd_gbps = 0.0;
    double nv_bw_utilization_pct = 0.0;      // (effective / theoretical) * 100
    double amd_bw_utilization_pct = 0.0;     // (effective / theoretical) * 100

    // Categorized Intelligence Findings
    std::vector<CrossVendorFinding> portable_findings;
    std::vector<CrossVendorFinding> vendor_specific_findings;

    // Diagnoses from standard engine
    DiagnosisReport nv_diagnosis;
    DiagnosisReport amd_diagnosis;

    std::string summary;
};

// Main entry point for cross-vendor intelligence analysis.
// Takes normalized metrics from an NVIDIA run and an AMD run. Never throws.
[[nodiscard]] CrossVendorReport analyze_cross_vendor(
    const profiling::GpuProfileMetrics& nv_m,
    const profiling::GpuProfileMetrics& amd_m,
    const std::string& workload_label = "vecadd");

// Render machine-readable JSON representation.
[[nodiscard]] std::string cross_vendor_to_json(const CrossVendorReport& r);

// Render human-readable cross-vendor report.
[[nodiscard]] std::string format_cross_vendor_report(const CrossVendorReport& r);

}  // namespace drishti::diagnosis

#endif  // DRISHTI_DIAGNOSIS_CROSS_VENDOR_H
