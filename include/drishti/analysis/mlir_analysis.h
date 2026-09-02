#ifndef DRISHTI_ANALYSIS_MLIR_ANALYSIS_H
#define DRISHTI_ANALYSIS_MLIR_ANALYSIS_H

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "drishti/analysis/analysis.h"
#include "drishti/provenance/provenance.h"

namespace mlir {
class AsmState;
class Dialect;
class MLIRContext;
class Operation;
class Block;
class Region;
class ModuleOp;
}  // namespace mlir

namespace drishti::analysis {

struct OpStat {
    std::string name;
    uint64_t count = 0;
    uint64_t regions = 0;
    uint64_t nested_regions = 0;
    uint64_t results = 0;
    uint64_t operands = 0;
};

struct DialectStat {
    std::string name;
    uint64_t op_count = 0;
    uint64_t distinct_ops = 0;
    std::vector<std::string> op_names;
};

struct FunctionInfo {
    std::string name;
    uint64_t op_count = 0;
    uint64_t block_count = 0;
    uint64_t region_count = 0;
    uint64_t argument_count = 0;
    uint64_t result_count = 0;
};

struct LoopInfo {
    std::string kind;
    std::string location_hint;
    uint64_t body_blocks = 0;
    uint64_t body_ops = 0;
};

struct StructuralStats {
    uint64_t total_operations = 0;
    uint64_t total_regions = 0;
    uint64_t total_blocks = 0;
    uint64_t total_arguments = 0;
    uint64_t total_results = 0;
    uint64_t total_operands = 0;

    uint64_t function_count = 0;
    uint64_t loop_count = 0;
    uint64_t branch_count = 0;

    std::vector<FunctionInfo> functions;
    std::vector<LoopInfo> loops;
    std::vector<OpStat> top_ops;
    std::vector<DialectStat> dialects;
    std::map<std::string, uint64_t> op_histogram;
};

class MlirAnalysisContext {
public:
    MlirAnalysisContext();
    ~MlirAnalysisContext();
    MlirAnalysisContext(const MlirAnalysisContext&) = delete;
    MlirAnalysisContext& operator=(const MlirAnalysisContext&) = delete;

    [[nodiscard]] mlir::MLIRContext& context() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MlirAnalysisEngine {
public:
    explicit MlirAnalysisEngine(MlirAnalysisContext& ctx);

    [[nodiscard]] std::optional<StructuralStats> analyze_file(const std::string& path,
                                                              std::string* error_out = nullptr);
    [[nodiscard]] std::optional<StructuralStats> analyze_string(std::string_view source,
                                                                std::string* error_out = nullptr);

    [[nodiscard]] const StructuralStats& last_stats() const noexcept { return stats_; }
    [[nodiscard]] const provenance::ProvenanceGraph& provenance() const noexcept { return tracker_.graph(); }
    [[nodiscard]] provenance::ProvenanceGraph& provenance() noexcept { return tracker_.graph(); }

    void enable_provenance(bool enable = true) { track_provenance_ = enable; }
    [[nodiscard]] bool provenance_enabled() const noexcept { return track_provenance_; }

    void set_pass_pipeline(const std::string& pipeline) { pass_pipeline_ = pipeline; }

private:
    MlirAnalysisContext* ctx_;
    StructuralStats stats_{};
    provenance::ProvenanceTracker tracker_{};
    bool track_provenance_ = true;
    std::unordered_map<mlir::Operation*, std::uint64_t> op_to_node_id_;
    std::map<std::uint64_t, std::uint64_t> uid_to_node_id_;
    std::string pass_pipeline_;
    std::uint64_t next_uid_ = 1;

    [[nodiscard]] std::optional<StructuralStats> analyze_impl(std::string_view source,
                                                              std::string_view buffer_name,
                                                              std::string* error_out);
    void walk(mlir::ModuleOp module);
    void record_stats(mlir::Operation* op);
    void record_dialect(const mlir::Dialect* dialect, mlir::Operation* op);
    void record_op(mlir::Operation* op);
    void record_function(mlir::Operation* op);
    void record_loop_if_detected(mlir::Operation* op);
    void record_branch_if_detected(mlir::Operation* op);
    [[nodiscard]] static std::string op_name(mlir::Operation* op);
    [[nodiscard]] static std::string dialect_name(mlir::Operation* op);
    [[nodiscard]] static std::string location_hint(mlir::Operation* op);
    [[nodiscard]] static bool is_loop_like(mlir::Operation* op);
    [[nodiscard]] static bool is_branch_like(mlir::Operation* op);
    [[nodiscard]] static uint64_t count_ops_in_region(mlir::Region& region);
    [[nodiscard]] static uint64_t count_blocks_in_region(mlir::Region& region);
    void finalize();

    void record_provenance(mlir::Operation* op);
    void compute_pipeline_transformations(std::uint64_t pass_id, mlir::ModuleOp module);
    [[nodiscard]] provenance::SourceLocation mlir_location_to_source(mlir::Operation* op) const;
    [[nodiscard]] std::vector<std::uint64_t> get_operand_ids(mlir::Operation* op) const;
    [[nodiscard]] std::vector<std::uint64_t> get_result_ids(mlir::Operation* op) const;
};

std::string format_report(const StructuralStats& stats, std::string_view source_name = "<input>");

}  // namespace drishti::analysis

#endif
