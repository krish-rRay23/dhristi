// Phase 11 tests: AMD ROCm / HIP backend interface, metric normalization, and mock execution.

#include "drishti/backends/rocm/rocm_backend.h"
#include "drishti/backends/backends.h"
#include "drishti/profiling/gpu_metrics.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using drishti::backends::BackendRegistry;
using drishti::backends::BackendKind;
using drishti::backends::rocm::VecaddConfig;
using drishti::backends::rocm::VecaddVariantConfig;
using drishti::backends::rocm::HipLaunchConfig;
using drishti::backends::rocm::HipRunResult;
using drishti::backends::rocm::query_device;
using drishti::backends::rocm::run_vecadd_profile;
using drishti::backends::rocm::run_vecadd_variant;
using drishti::backends::rocm::run_hip_kernel;
using drishti::backends::rocm::set_mock_mode;

TEST(RocmBackend, BackendRegistryIncludesRocm) {
    BackendRegistry reg;
    bool found_rocm = false;
    for (const auto& b : reg.list()) {
        if (b.kind == BackendKind::ROCm) {
            found_rocm = true;
            EXPECT_EQ(b.name, "rocm");
        }
    }
    EXPECT_TRUE(found_rocm);
}

TEST(RocmBackend, MockDeviceQueryProducesNormalizedMetrics) {
    set_mock_mode(true);
    drishti::profiling::GpuDeviceModel dev;
    const bool ok = query_device(dev);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(dev.present);
    EXPECT_EQ(dev.backend, "rocm");
    EXPECT_NE(dev.name.find("AMD"), std::string::npos);
    EXPECT_EQ(dev.compute_major, 11);
    EXPECT_EQ(dev.compute_minor, 0);
    EXPECT_EQ(dev.sm_count, 84);  // 84 Compute Units
    EXPECT_EQ(dev.max_threads_per_block, 1024);
    EXPECT_GT(dev.total_mem_bytes, 0u);
}

TEST(RocmBackend, RunVecaddProfileEmitsNormalizedMetrics) {
    set_mock_mode(true);
    VecaddConfig cfg;
    cfg.num_elements = 65536;
    cfg.block_size = 256;
    cfg.repeats = 5;

    drishti::profiling::GpuProfileMetrics m;
    std::string err;
    const bool ok = run_vecadd_profile(cfg, m, &err);
    EXPECT_TRUE(ok) << err;
    EXPECT_TRUE(m.ok);
    EXPECT_EQ(m.device.backend, "rocm");
    EXPECT_EQ(m.num_elements, 65536u);
    EXPECT_EQ(m.block_size, 256);
    EXPECT_EQ(m.grid_size, 256);
    EXPECT_GT(m.kernel_ms_min, 0.0);
    EXPECT_GT(m.gbps_effective, 0.0);
    EXPECT_TRUE(m.correct);
}

TEST(RocmBackend, RunVecaddVariantRecopyScalesMemoryTraffic) {
    set_mock_mode(true);
    VecaddVariantConfig cfg_base;
    cfg_base.num_elements = 65536;
    cfg_base.repeats = 5;
    cfg_base.recopy_per_launch = false;

    drishti::profiling::GpuProfileMetrics m_base;
    std::string err;
    ASSERT_TRUE(run_vecadd_variant(cfg_base, m_base, &err));

    VecaddVariantConfig cfg_recopy = cfg_base;
    cfg_recopy.recopy_per_launch = true;
    drishti::profiling::GpuProfileMetrics m_recopy;
    ASSERT_TRUE(run_vecadd_variant(cfg_recopy, m_recopy, &err));

    EXPECT_EQ(m_recopy.bytes_moved, m_base.bytes_moved * 5u);
    EXPECT_GT(m_recopy.h2d_ms, m_base.h2d_ms);
}

TEST(RocmBackend, RunHipKernelExecutesLaunchSequence) {
    set_mock_mode(true);
    HipLaunchConfig cfg;
    cfg.num_elements = 65536;
    cfg.block_size = 256;
    cfg.repeats = 3;
    cfg.kernel_label = "fusedemo_hip";
    cfg.launches = {
        {"_fusedemo_kernel_0", {}, {0, 1, 2}},
        {"_fusedemo_kernel_1", {}, {2, 3, 4}}
    };

    HipRunResult res;
    std::string err;
    const bool ok = run_hip_kernel(cfg, res, &err);
    EXPECT_TRUE(ok) << err;
    EXPECT_TRUE(res.metrics.ok);
    EXPECT_EQ(res.metrics.device.backend, "rocm");
    EXPECT_EQ(res.metrics.kernel, "fusedemo_hip");
    EXPECT_EQ(res.per_kernel.size(), 2u);
    EXPECT_EQ(res.per_kernel[0].entry, "_fusedemo_kernel_0");
    EXPECT_EQ(res.per_kernel[1].entry, "_fusedemo_kernel_1");
}

}  // namespace
