// Focused Phase 9 tests: real MLIR-to-GPU transformation experiment.
//
// Lowering tests are host-side (PTX codegen needs no GPU) and run wherever
// MLIR is available. Execution tests run the full baseline-vs-fused flow on
// the RTX 3050 and check the verdict machinery, not fixed timings.

#include "drishti/optimizer/compiler_experiment.h"

#include "drishti/analysis/gpu_lowering.h"
#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/core/version.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using drishti::optimizer::CompilerExperimentConfig;
using drishti::optimizer::compiler_experiment_to_json;
using drishti::optimizer::compiler_experiment_to_pass_info;
using drishti::optimizer::format_compiler_experiment_report;
using drishti::optimizer::run_compiler_experiment;
using drishti::optimizer::Verdict;

bool have_toolchain() {
    return drishti::core::have_mlir() && drishti::core::have_cuda() &&
           drishti::backends::cuda::device_present();
}

TEST(CompilerLowering, PipelinesDifferByOneTransformation) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    const std::string base = drishti::analysis::fusion_baseline_pipeline();
    const std::string fused = drishti::analysis::fusion_transformed_pipeline();
    EXPECT_NE(fused.find("affine-loop-fusion"), std::string::npos);
    EXPECT_EQ(base.find("affine-loop-fusion"), std::string::npos);
    EXPECT_NE(base.find("convert-affine-for-to-gpu"), std::string::npos);
    EXPECT_NE(fused.find("convert-affine-for-to-gpu"), std::string::npos);
    EXPECT_NE(base.find("convert-gpu-to-nvvm"), std::string::npos);
}

TEST(CompilerLowering, WorkloadHasTwoNestedLoops) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    const std::string src = drishti::analysis::fusion_workload_mlir(65536);
    EXPECT_NE(src.find("func.func @fusedemo"), std::string::npos);
    EXPECT_NE(src.find("memref<256x256xf32>"), std::string::npos);
    EXPECT_NE(src.find("arith.addf"), std::string::npos);
    EXPECT_NE(src.find("arith.mulf"), std::string::npos);
    // Two perfectly nested 2-deep loop nests: producer then consumer.
    std::size_t count = 0, pos = 0;
    while ((pos = src.find("affine.for", pos)) != std::string::npos) {
        ++count;
        ++pos;
    }
    EXPECT_EQ(count, 4u);
}

TEST(CompilerLowering, BaselineLowersToTwoKernels) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    std::string err;
    const auto out = drishti::analysis::lower_to_ptx(
        drishti::analysis::fusion_workload_mlir(4096),
        drishti::analysis::fusion_baseline_pipeline(), 8, 6, &err);
    ASSERT_TRUE(out.ok) << err;
    EXPECT_EQ(out.num_kernels(), 2u);
    EXPECT_FALSE(out.ptx.empty());
    EXPECT_TRUE(out.ptx.find("ld.") != std::string::npos);
    EXPECT_TRUE(out.ptx.find("st.") != std::string::npos);
    ASSERT_EQ(out.launches.size(), 2u);
    // Compiler-described launch signatures: (a,b,t) then (t,c,d).
    EXPECT_EQ(out.launches[0].buffer_slots, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(out.launches[1].buffer_slots, (std::vector<int>{2, 3, 4}));
    EXPECT_FALSE(out.launches[0].entry.empty());
    EXPECT_NE(out.launches[0].entry, out.launches[1].entry);
}

TEST(CompilerLowering, FusedLowersToOneKernel) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    std::string err;
    const auto out = drishti::analysis::lower_to_ptx(
        drishti::analysis::fusion_workload_mlir(4096),
        drishti::analysis::fusion_transformed_pipeline(), 8, 6, &err);
    ASSERT_TRUE(out.ok) << err;
    ASSERT_EQ(out.num_kernels(), 1u);
    EXPECT_FALSE(out.entry_names.front().empty());
    ASSERT_EQ(out.launches.size(), 1u);
    EXPECT_EQ(out.launches[0].buffer_slots, (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(CompilerLowering, BadSourceFailsCleanly) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    std::string err;
    const auto out = drishti::analysis::lower_to_ptx(
        "module { this is not mlir }", drishti::analysis::fusion_baseline_pipeline(),
        8, 6, &err);
    EXPECT_FALSE(out.ok);
    EXPECT_FALSE(err.empty());
}

TEST(CompilerExperiment, RejectsBadGeometryWithoutGpu) {
    CompilerExperimentConfig cfg;
    cfg.num_elements = 1000;  // not a perfect square
    cfg.block_size = 256;
    std::string err;
    const auto r = run_compiler_experiment(cfg, &err);
    EXPECT_FALSE(r.ok);
    EXPECT_NE(err.find("perfect square"), std::string::npos);
}

TEST(CompilerExperiment, RejectsMismatchedBlockWithoutGpu) {
    CompilerExperimentConfig cfg;
    cfg.num_elements = 65536;
    cfg.block_size = 128;  // compiler-fixed geometry is 256x256
    std::string err;
    const auto r = run_compiler_experiment(cfg, &err);
    EXPECT_FALSE(r.ok);
    EXPECT_NE(err.find("must equal dim"), std::string::npos);
}

TEST(CompilerExperiment, LiveRtx3050BeforeAfter) {
    if (!have_toolchain()) GTEST_SKIP() << "MLIR/CUDA toolchain unavailable";
    CompilerExperimentConfig cfg;
    // 65536 elements (256x256): each array is 256KB, placing kernel timing
    // in the stable 0.1ms+ range where CUDA event resolution is reliable.
    // block_size must equal dim = sqrt(num_elements) = 256 per the workload
    // geometry constraint.
    cfg.num_elements = 65536;
    cfg.block_size = 256;
    cfg.repeats = 5;
    std::string err;
    const auto r = run_compiler_experiment(cfg, &err);
    ASSERT_TRUE(r.ok) << err;
    // Chain anchors present on both sides (1 func + 4/2 loops).
    EXPECT_FALSE(r.baseline.func_anchor.empty());
    EXPECT_FALSE(r.candidate.func_anchor.empty());
    EXPECT_EQ(r.baseline.loop_anchors.size(), 4u);
    EXPECT_EQ(r.candidate.loop_anchors.size(), 2u);
    EXPECT_EQ(r.baseline.kernels.size(), 2u);
    EXPECT_EQ(r.candidate.kernels.size(), 1u);
    EXPECT_EQ(r.baseline.metrics.num_elements, 65536u);
    EXPECT_EQ(r.candidate.metrics.num_elements, 65536u);
    EXPECT_TRUE(r.baseline.metrics.correct);
    EXPECT_TRUE(r.candidate.metrics.correct);
    ASSERT_EQ(r.baseline.per_kernel.size(), 2u);
    ASSERT_EQ(r.candidate.per_kernel.size(), 1u);
    // Verdict is a valid classification of measured values.
    EXPECT_TRUE(r.result.verdict == Verdict::Improved ||
                r.result.verdict == Verdict::Unchanged ||
                r.result.verdict == Verdict::Regressed);
    EXPECT_EQ(r.result.experiment_id, "compiler-fusion");
    EXPECT_GT(r.result.baseline_value, 0.0);
    EXPECT_GT(r.result.candidate_value, 0.0);
    // Structural: one fused launch cannot cost much more than two launches.
    EXPECT_LE(r.result.candidate_value, r.result.baseline_value * 3.0);
    const std::string j = compiler_experiment_to_json(r);
    for (const char* key : {"drishti.compiler_experiment/v1", "affine-loop-fusion",
                            "kernels", "loop_anchors", "kernel-min-ms", "verdict",
                            "per_kernel_ms_min"}) {
        EXPECT_NE(j.find(key), std::string::npos) << "missing key: " << key;
    }
    const std::string rep = format_compiler_experiment_report(r);
    EXPECT_NE(rep.find("Drishti Compiler Experiment"), std::string::npos);
    EXPECT_NE(rep.find("Correctness"), std::string::npos);
    const auto pass = compiler_experiment_to_pass_info(r);
    EXPECT_EQ(pass.name, "cexperiment:fusion");
}

}  // namespace
