#include "drishti/triton/triton_integration.h"

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/core/version.h"

#include <gtest/gtest.h>
#include <string>

namespace {

using drishti::triton::TritonWorkloadConfig;
using drishti::triton::run_triton_pipeline;
using drishti::triton::triton_pipeline_to_json;
using drishti::triton::format_triton_pipeline_report;

TEST(TritonIntegration, FusedAddReluCompilationArtifacts) {
    TritonWorkloadConfig cfg;
    cfg.workload_name = "fused_add_relu";
    cfg.num_elements = 65536;
    cfg.block_size = 256;
    cfg.verify_gpu = false;

    std::string err;
    auto rep = run_triton_pipeline(cfg, &err);

    ASSERT_TRUE(rep.ok) << "Triton compilation failed: " << rep.error;
    EXPECT_EQ(rep.workload_name, "fused_add_relu");
    EXPECT_EQ(rep.artifacts.kernel_name, "fused_add_relu_kernel");

    // TTIR checks
    EXPECT_FALSE(rep.artifacts.ttir.empty());
    EXPECT_NE(rep.artifacts.ttir.find("tt.func"), std::string::npos);

    // TTGIR checks
    EXPECT_FALSE(rep.artifacts.ttgir.empty());
    EXPECT_NE(rep.artifacts.ttgir.find("#ttg.blocked"), std::string::npos);

    // LLVM IR checks
    EXPECT_FALSE(rep.artifacts.llvm_ir.empty());
    EXPECT_NE(rep.artifacts.llvm_ir.find("LLVMDialectModule"), std::string::npos);

    // PTX checks
    EXPECT_FALSE(rep.artifacts.ptx.empty());
    EXPECT_NE(rep.artifacts.ptx.find("fused_add_relu_kernel"), std::string::npos);
    EXPECT_NE(rep.artifacts.ptx.find(".target sm_86"), std::string::npos);
}

TEST(TritonIntegration, VectorAddCompilationArtifacts) {
    TritonWorkloadConfig cfg;
    cfg.workload_name = "vector_add";
    cfg.num_elements = 32768;
    cfg.block_size = 256;
    cfg.verify_gpu = false;

    std::string err;
    auto rep = run_triton_pipeline(cfg, &err);

    ASSERT_TRUE(rep.ok) << "Triton compilation failed: " << rep.error;
    EXPECT_EQ(rep.artifacts.kernel_name, "vector_add_kernel");
    EXPECT_FALSE(rep.artifacts.ptx.empty());
    EXPECT_NE(rep.artifacts.ptx.find("vector_add_kernel"), std::string::npos);
}

TEST(TritonIntegration, ProvenanceGraphStructure) {
    TritonWorkloadConfig cfg;
    cfg.workload_name = "fused_add_relu";
    cfg.num_elements = 65536;
    cfg.verify_gpu = false;

    auto rep = run_triton_pipeline(cfg);
    ASSERT_TRUE(rep.ok);

    const auto& g = rep.provenance;
    EXPECT_EQ(g.node_count(), 6u);
    EXPECT_EQ(g.edge_count(), 5u);
    EXPECT_EQ(rep.workload_name, "fused_add_relu");
}

TEST(TritonIntegration, CostModelAndDiagnosis) {
    TritonWorkloadConfig cfg;
    cfg.workload_name = "fused_add_relu";
    cfg.num_elements = 65536;
    cfg.verify_gpu = false;

    auto rep = run_triton_pipeline(cfg);
    ASSERT_TRUE(rep.ok);

    EXPECT_GT(rep.cost_estimate.predicted_kernel_ms, 0.0);
    EXPECT_FALSE(rep.cost_estimate.bottleneck_regime.empty());
    EXPECT_GT(rep.cost_estimate.theoretical_occupancy_pct, 0.0);
    EXPECT_TRUE(rep.diagnosis.ok);
}

TEST(TritonIntegration, LiveGPUExecution) {
    if (!drishti::core::have_cuda() || !drishti::backends::cuda::device_present()) {
        GTEST_SKIP() << "CUDA GPU unavailable on this system";
    }

    TritonWorkloadConfig cfg;
    cfg.workload_name = "fused_add_relu";
    // 1M elements so CUDA-event time is bandwidth-dominated, not WDDM
    // dispatch latency (~40us), which the cost-model bound assumes.
    cfg.num_elements = 1u << 20;
    cfg.block_size = 256;
    cfg.repeats = 5;
    cfg.verify_gpu = true;

    std::string err;
    auto rep = run_triton_pipeline(cfg, &err);

    ASSERT_TRUE(rep.ok) << "Triton pipeline failed: " << rep.error;
    EXPECT_TRUE(rep.gpu_executed);
    EXPECT_TRUE(rep.gpu_correct) << "GPU result mismatch: " << rep.error;
    EXPECT_GT(rep.measured_kernel_ms, 0.0);
    EXPECT_GT(rep.gpu_metrics.gbps_effective, 0.0);
    EXPECT_GT(rep.measured_tflops, 0.0);

    EXPECT_GT(rep.cost_estimate.predicted_kernel_ms, 0.0);
    EXPECT_LE(rep.cost_validation.error_percent, 75.0);
}

TEST(TritonIntegration, JsonAndReportFormatting) {
    TritonWorkloadConfig cfg;
    cfg.workload_name = "fused_add_relu";
    cfg.num_elements = 65536;
    cfg.verify_gpu = false;

    auto rep = run_triton_pipeline(cfg);
    ASSERT_TRUE(rep.ok);

    std::string json = triton_pipeline_to_json(rep);
    EXPECT_NE(json.find("fused_add_relu"), std::string::npos);
    EXPECT_NE(json.find("artifacts"), std::string::npos);

    std::string report = format_triton_pipeline_report(rep, true);
    EXPECT_NE(report.find("PHASE 18: DEEP TRITON INTEGRATION REPORT"), std::string::npos);
    EXPECT_NE(report.find("COMPILER STAGES CAPTURED"), std::string::npos);
    EXPECT_NE(report.find("[CAPTURED TRITON-IR (TTIR)]"), std::string::npos);
}

TEST(TritonIntegration, VectorizationCodegenCaseStudy) {
    const bool have_gpu = drishti::core::have_cuda() && drishti::backends::cuda::device_present();

    // 1. Run unvectorized scalar workload
    TritonWorkloadConfig cfg_scalar;
    cfg_scalar.workload_name = "vector_add_scalar";
    cfg_scalar.num_elements = 65536;
    cfg_scalar.block_size = 1024;
    cfg_scalar.verify_gpu = have_gpu;

    auto rep_scalar = run_triton_pipeline(cfg_scalar);
    ASSERT_TRUE(rep_scalar.ok);
    if (have_gpu) {
        EXPECT_TRUE(rep_scalar.gpu_executed);
        EXPECT_TRUE(rep_scalar.gpu_correct);
    }

    // Verify scalar PTX instructions and TTGIR sizePerThread = [1]
    EXPECT_NE(rep_scalar.artifacts.ttgir.find("sizePerThread = [1]"), std::string::npos);
    EXPECT_NE(rep_scalar.artifacts.ptx.find("ld.global.b32"), std::string::npos);

    // 2. Run 128-bit vectorized workload
    TritonWorkloadConfig cfg_vec;
    cfg_vec.workload_name = "vector_add_vectorized";
    cfg_vec.num_elements = 65536;
    cfg_vec.block_size = 1024;
    cfg_vec.verify_gpu = have_gpu;

    auto rep_vec = run_triton_pipeline(cfg_vec);
    ASSERT_TRUE(rep_vec.ok);
    if (have_gpu) {
        EXPECT_TRUE(rep_vec.gpu_executed);
        EXPECT_TRUE(rep_vec.gpu_correct);
    }

    // Verify vectorized PTX instructions and TTGIR sizePerThread = [4]
    EXPECT_NE(rep_vec.artifacts.ttgir.find("sizePerThread = [4]"), std::string::npos);
    EXPECT_NE(rep_vec.artifacts.ptx.find("ld.global.v4.b32"), std::string::npos);
}

}  // namespace
