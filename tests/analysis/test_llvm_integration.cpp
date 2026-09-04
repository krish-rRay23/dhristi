#include <gtest/gtest.h>

#include "drishti/analysis/llvm_integration.h"
#include "drishti/core/config.h"

namespace drishti::analysis {

TEST(LlvmIntegrationTest, RealMlirToLlvmLowering) {
    LlvmPipelineConfig cfg;
    cfg.workload_label = "fusion";
    cfg.num_elements = 65536;
    cfg.llvm_passes = {"sroa", "instcombine", "simplifycfg", "dce"};
    cfg.verify_gpu = true;

    auto rep = run_llvm_pipeline(cfg);

    EXPECT_TRUE(rep.ok) << "Pipeline error: " << rep.error;
    EXPECT_FALSE(rep.raw_llvm_ir.empty());
    EXPECT_FALSE(rep.optimized_llvm_ir.empty());
    EXPECT_GT(rep.raw_stats.instruction_count, 0u);
    EXPECT_GT(rep.optimized_stats.instruction_count, 0u);

    // Verify raw LLVM IR contains expected kernel definitions and target triple
    EXPECT_NE(rep.raw_llvm_ir.find("target triple = \"nvptx64-nvidia-cuda\""), std::string::npos);
    EXPECT_NE(rep.raw_llvm_ir.find("void @"), std::string::npos);

    // Verify PTX codegen
    EXPECT_FALSE(rep.lowered_ptx.empty());
    EXPECT_NE(rep.lowered_ptx.find(".entry"), std::string::npos);
}

TEST(LlvmIntegrationTest, RealLlvmPassesExecution) {
    LlvmPipelineConfig cfg;
    cfg.workload_label = "fusion";
    cfg.num_elements = 65536;
    cfg.llvm_passes = {"sroa", "instcombine", "simplifycfg", "dce"};
    cfg.verify_gpu = false;

    auto rep = run_llvm_pipeline(cfg);

    EXPECT_TRUE(rep.ok);
    EXPECT_EQ(rep.passes_applied.size(), 4u);

    // Verify each pass was executed and recorded
    EXPECT_EQ(rep.passes_applied[0].pass_name, "sroa");
    EXPECT_EQ(rep.passes_applied[1].pass_name, "instcombine");
    EXPECT_EQ(rep.passes_applied[2].pass_name, "simplifycfg");
    EXPECT_EQ(rep.passes_applied[3].pass_name, "dce");

    // Verify instruction optimization occurred (optimized <= raw)
    EXPECT_LE(rep.optimized_stats.instruction_count, rep.raw_stats.instruction_count);
    EXPECT_LE(rep.total_instruction_delta, 0);
}

TEST(LlvmIntegrationTest, ProvenanceTrackingAcrossMlirAndLlvm) {
    LlvmPipelineConfig cfg;
    cfg.workload_label = "fusion";
    cfg.num_elements = 65536;
    cfg.llvm_passes = {"sroa", "instcombine", "dce"};
    cfg.verify_gpu = false;

    auto rep = run_llvm_pipeline(cfg);

    EXPECT_TRUE(rep.ok);
    EXPECT_GE(rep.provenance.nodes().size(), 4u);
    EXPECT_GE(rep.provenance.passes().size(), 3u);
    EXPECT_GE(rep.provenance.edges().size(), 2u);

    // Verify pass info conversion
    auto pinfo = llvm_pipeline_to_pass_info(rep);
    EXPECT_EQ(pinfo.name, "llvm-deep-integration");
}

TEST(LlvmIntegrationTest, JsonAndFormattedReporting) {
    LlvmPipelineConfig cfg;
    cfg.workload_label = "fusion";
    cfg.num_elements = 65536;
    cfg.verify_gpu = true;

    auto rep = run_llvm_pipeline(cfg);

    const std::string json = llvm_pipeline_to_json(rep);
    EXPECT_NE(json.find("\"schema\": \"drishti.llvm/v1\""), std::string::npos);
    EXPECT_NE(json.find("\"raw_llvm_stats\""), std::string::npos);
    EXPECT_NE(json.find("\"optimized_llvm_stats\""), std::string::npos);

    const std::string report = format_llvm_pipeline_report(rep, true);
    EXPECT_NE(report.find("PHASE 17: DEEP LLVM INTEGRATION REPORT"), std::string::npos);
    EXPECT_NE(report.find("LLVM Optimization Pass Execution Chain"), std::string::npos);
    EXPECT_NE(report.find("Raw Lowered LLVM IR"), std::string::npos);
}

}  // namespace drishti::analysis
