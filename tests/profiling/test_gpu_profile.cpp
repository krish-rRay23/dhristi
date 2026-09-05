// Focused Phase 4 tests: NVIDIA/CUDA profiling backend + metric model.
//
// Tests that need real hardware skip cleanly when no CUDA device/driver is
// present; the metric-model tests always run.

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/profiling/gpu_metrics.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using drishti::backends::cuda::cupti_status;
using drishti::backends::cuda::device_present;
using drishti::backends::cuda::query_device;
using drishti::backends::cuda::run_vecadd_profile;
using drishti::backends::cuda::VecaddConfig;
using drishti::profiling::format_gpu_report;
using drishti::profiling::gpu_metrics_to_json;
using drishti::profiling::gpu_metrics_to_pass_info;
using drishti::profiling::GpuProfileMetrics;

TEST(GpuMetricsModel, JsonContainsRequiredFields) {
    GpuProfileMetrics m;
    m.ok = true;
    m.device.backend = "cuda";
    m.device.present = true;
    m.device.name = "NVIDIA GeForce RTX 3050 Laptop GPU";
    m.kernel = "vecadd";
    m.num_elements = 1024;
    m.kernel_ms_avg = 0.0123;
    m.correct = true;
    m.timing_source = "cuda-events";

    const std::string j = gpu_metrics_to_json(m);
    for (const char* key :
         {"drishti.gpu_profile/v1", "vecadd", "kernel_ms_avg", "gbps_effective",
          "cupti_available", "timing_source", "compute_capability", "correct"}) {
        EXPECT_NE(j.find(key), std::string::npos) << "missing key: " << key;
    }
}

TEST(GpuMetricsModel, ProvenancePassWiring) {
    GpuProfileMetrics m;
    m.ok = true;
    m.kernel = "vecadd";
    m.num_elements = 4096;
    m.kernel_ms_avg = 0.05;
    m.gbps_effective = 120.5;
    m.correct = true;
    m.device.backend = "cuda";
    m.device.name = "NVIDIA GeForce RTX 3050 Laptop GPU";

    const auto pass = gpu_metrics_to_pass_info(m);
    EXPECT_EQ(pass.name, "cuda-profile:vecadd");
    EXPECT_NE(pass.description.find("ok"), std::string::npos);
    EXPECT_NE(pass.description.find("RTX 3050"), std::string::npos);
    EXPECT_TRUE(pass.enabled_by_default);
}

TEST(GpuMetricsModel, FailureReportMentionsError) {
    GpuProfileMetrics m;
    m.ok = false;
    m.error = "no CUDA-capable device found";
    const std::string r = format_gpu_report(m);
    EXPECT_NE(r.find("FAILED"), std::string::npos);
    EXPECT_NE(r.find("no CUDA-capable device"), std::string::npos);
}

// Keep detection assertions in terms of the public device model.
TEST(CudaBackend, DetectionMatchesQuery) {
    drishti::profiling::GpuDeviceModel dev;
    const bool ok = query_device(dev);
    EXPECT_EQ(ok, device_present());
    EXPECT_EQ(dev.present, ok);
    if (ok) {
        EXPECT_FALSE(dev.name.empty());
        EXPECT_GT(dev.total_mem_bytes, 0u);
        EXPECT_GT(dev.sm_count, 0);
    } else {
        EXPECT_FALSE(dev.error.empty());
    }
}

TEST(CudaBackend, CuptiProbeNeverFails) {
    bool available = true;
    std::string version;
    cupti_status(available, version);
    EXPECT_FALSE(version.empty());
    if (!available) {
        // Reason must mention the missing toolkit/CUPTI library.
        const bool explains = version.find("CUPTI") != std::string::npos ||
                              version.find("toolkit") != std::string::npos;
        EXPECT_TRUE(explains) << version;
    }
}

TEST(CudaBackend, VecaddSmallProfile) {
    if (!device_present()) GTEST_SKIP() << "no CUDA device/driver present";
    VecaddConfig cfg;
    cfg.num_elements = 4096;
    cfg.block_size = 256;
    cfg.repeats = 2;
    GpuProfileMetrics out;
    std::string err;
    ASSERT_TRUE(run_vecadd_profile(cfg, out, &err)) << err;
    EXPECT_TRUE(out.ok);
    EXPECT_TRUE(out.correct);
    EXPECT_GT(out.kernel_ms_avg, 0.0);
    EXPECT_GT(out.kernel_ms_min, 0.0);
    EXPECT_GT(out.gbps_effective, 0.0);
    EXPECT_EQ(out.num_elements, 4096u);
    EXPECT_EQ(out.kernel, "vecadd");
}

TEST(CudaBackend, VecaddRejectsBadConfig) {
    VecaddConfig cfg;
    cfg.num_elements = 0;
    GpuProfileMetrics out;
    std::string err;
    EXPECT_FALSE(run_vecadd_profile(cfg, out, &err));
    EXPECT_FALSE(err.empty());
}

}  // namespace
