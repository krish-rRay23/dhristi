// Phase 12 tests: Cross-Vendor Performance Intelligence & Metric Normalization.

#include "drishti/diagnosis/cross_vendor.h"
#include "drishti/backends/rocm/rocm_backend.h"
#include "drishti/profiling/gpu_metrics.h"

#include <gtest/gtest.h>
#include <string>

namespace {

using drishti::diagnosis::CrossVendorReport;
using drishti::diagnosis::analyze_cross_vendor;
using drishti::diagnosis::cross_vendor_to_json;
using drishti::diagnosis::format_cross_vendor_report;
using drishti::profiling::GpuDeviceModel;
using drishti::profiling::GpuProfileMetrics;

TEST(CrossVendorDiagnosis, GeneratesValidReportFromProfiles) {
    // 1. Setup NVIDIA GPU Profile
    GpuProfileMetrics nv_m;
    nv_m.ok = true;
    nv_m.device.present = true;
    nv_m.device.backend = "cuda";
    nv_m.device.name = "NVIDIA GeForce RTX 3050 Laptop GPU";
    nv_m.device.compute_major = 8;
    nv_m.device.compute_minor = 6;
    nv_m.device.sm_count = 16;
    nv_m.device.mem_bus_width_bits = 128;
    nv_m.device.mem_theoretical_gbps = 192.0;
    nv_m.device.total_mem_bytes = 4294443008ULL;
    nv_m.device.driver_version = "592.27";
    nv_m.kernel = "vecadd";
    nv_m.num_elements = 1048576;
    nv_m.block_size = 256;
    nv_m.grid_size = 4096;
    nv_m.repeats = 10;
    nv_m.kernel_ms_avg = 0.0998;
    nv_m.kernel_ms_min = 0.0788;
    nv_m.h2d_ms = 1.2646;
    nv_m.d2h_ms = 0.7811;
    nv_m.wall_ms = 80.9438;
    nv_m.gbps_effective = 126.0995;
    nv_m.bytes_moved = 12582912;
    nv_m.correct = true;
    nv_m.timing_source = "cuda-events";

    // 2. Setup AMD ROCm Mock GPU Profile
    drishti::backends::rocm::set_mock_mode(true);
    GpuProfileMetrics amd_m;
    std::string err;
    drishti::backends::rocm::VecaddConfig cfg;
    cfg.num_elements = 1048576;
    cfg.block_size = 256;
    cfg.repeats = 10;
    ASSERT_TRUE(drishti::backends::rocm::run_vecadd_profile(cfg, amd_m, &err));

    // 3. Analyze Cross-Vendor Intelligence
    CrossVendorReport rep = analyze_cross_vendor(nv_m, amd_m, "vecadd_benchmark");
    EXPECT_TRUE(rep.ok);
    EXPECT_EQ(rep.workload, "vecadd_benchmark");
    EXPECT_FALSE(rep.nv_is_mock);
    EXPECT_TRUE(rep.amd_is_mock);  // Mock AMD GPU must be explicitly flagged

    // Latency & Bandwidth comparisons
    EXPECT_GT(rep.latency_ratio_nv_vs_amd, 0.0);
    EXPECT_GT(rep.bandwidth_nv_gbps, 0.0);
    EXPECT_GT(rep.bandwidth_amd_gbps, 0.0);

    // Portable vs Vendor-Specific Categorization
    EXPECT_FALSE(rep.portable_findings.empty());
    EXPECT_FALSE(rep.vendor_specific_findings.empty());

    bool found_regime = false;
    for (const auto& f : rep.portable_findings) {
        if (f.title.find("Memory-Bound") != std::string::npos) {
            found_regime = true;
            EXPECT_TRUE(f.is_portable);
        }
    }
    EXPECT_TRUE(found_regime);

    bool found_arch = false;
    for (const auto& f : rep.vendor_specific_findings) {
        if (f.title.find("Multiprocessor") != std::string::npos) {
            found_arch = true;
            EXPECT_FALSE(f.is_portable);
            EXPECT_NE(f.amd_trait.find("[MOCK"), std::string::npos);
        }
    }
    EXPECT_TRUE(found_arch);

    // JSON and String Formatting
    std::string json = cross_vendor_to_json(rep);
    EXPECT_NE(json.find("drishti.cross_vendor/v1"), std::string::npos);

    std::string formatted = format_cross_vendor_report(rep);
    EXPECT_NE(formatted.find("Phase 12"), std::string::npos);
    EXPECT_NE(formatted.find("UNVERIFIED ON PHYSICAL AMD SILICON"), std::string::npos);
}

}  // namespace
