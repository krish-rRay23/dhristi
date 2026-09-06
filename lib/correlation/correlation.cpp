#include "drishti/correlation/correlation.h"

#include <iomanip>
#include <sstream>

#include "drishti/core/config.h"

#if DRISHTI_HAVE_MLIR
#include "drishti/analysis/mlir_analysis.h"
#endif
#if DRISHTI_HAVE_CUDA
#include "drishti/backends/cuda/cuda_backend.h"
#endif

namespace drishti::correlation {
namespace {

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

std::string dbl(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << v;
    return oss.str();
}

std::string loc_string(const provenance::SourceLocation& loc) {
    std::ostringstream oss;
    oss << (loc.file.empty() ? "<input>" : loc.file) << ":" << loc.line << ":"
        << loc.column;
    return oss.str();
}

}  // namespace

std::string relation_name(RelationKind r) {
    switch (r) {
        case RelationKind::SameComputation: return "same-computation";
        case RelationKind::CoExecuted: return "co-executed";
    }
    return "unknown";
}

std::string generate_vecadd_mlir(std::size_t n) {
    std::ostringstream oss;
    oss << "module {\n"
        << "  func.func @vecadd(%a: memref<" << n << "xf32>, %b: memref<" << n
        << "xf32>, %c: memref<" << n << "xf32>) {\n"
        << "    %c0 = arith.constant 0 : index\n"
        << "    %cN = arith.constant " << n << " : index\n"
        << "    %c1 = arith.constant 1 : index\n"
        << "    scf.for %i = %c0 to %cN step %c1 {\n"
        << "      %x = memref.load %a[%i] : memref<" << n << "xf32>\n"
        << "      %y = memref.load %b[%i] : memref<" << n << "xf32>\n"
        << "      %z = arith.addf %x, %y : f32\n"
        << "      memref.store %z, %c[%i] : memref<" << n << "xf32>\n"
        << "    }\n"
        << "    return\n"
        << "  }\n"
        << "}\n";
    return oss.str();
}

CorrelationRecord run_correlation(const CorrelationConfig& cfg, std::string* err) {
    CorrelationRecord r;
    r.pipeline = cfg.pipeline;
    r.num_elements = cfg.num_elements;
    r.block_size = cfg.block_size;
    const auto fail = [&](const std::string& msg) {
        r.ok = false;
        r.error = msg;
        if (err) *err = msg;
        return r;
    };

#if !DRISHTI_HAVE_MLIR
    return fail("correlate requires the MLIR backend (built with DRISHTI_ENABLE_LLVM=OFF)");
#elif !DRISHTI_HAVE_CUDA
    return fail("correlate requires the CUDA backend (built with DRISHTI_ENABLE_CUDA=OFF)");
#else
    if (cfg.num_elements == 0 || cfg.num_elements > (1u << 28))
        return fail("num_elements out of range (1 .. 268435456)");

    const bool embedded = cfg.mlir_file.empty();
    const std::string mlir_src =
        embedded ? generate_vecadd_mlir(cfg.num_elements) : std::string{};
    r.mlir_source_label = embedded ? "<embedded-vecadd>" : cfg.mlir_file;
    r.kernel = "vecadd";

    // ---- Compiler stage: analyze MLIR with provenance + pass pipeline. ----
    analysis::MlirAnalysisContext ctx;
    analysis::MlirAnalysisEngine engine{ctx};
    engine.set_pass_pipeline(cfg.pipeline);
    std::string mlir_err;
    std::optional<analysis::StructuralStats> stats;
    if (embedded) {
        stats = engine.analyze_string(mlir_src, &mlir_err);
    } else {
        stats = engine.analyze_file(cfg.mlir_file, &mlir_err);
    }
    if (!stats) return fail("MLIR analysis failed: " + mlir_err);
    r.mlir_ops = stats->total_operations;
    r.mlir_funcs = stats->function_count;

    const provenance::ProvenanceGraph& graph = engine.provenance();
    r.transform_edges = graph.edge_count();
    if (!graph.passes().empty()) {
        // The pipeline pass is the last recorded pass.
        r.pipeline_pass_id = static_cast<std::uint64_t>(graph.passes().size());
    }

    // Anchors: function node + compute (add) node, by op name.
    bool has_vecadd_func = false;
    for (const auto& f : stats->functions) {
        if (f.name == "vecadd" || f.name == "@vecadd") {
            has_vecadd_func = true;
            break;
        }
    }
    const provenance::OperationNode* func_node = nullptr;
    const provenance::OperationNode* add_node = nullptr;
    for (const auto& node : graph.nodes()) {
        if (!func_node && node.op_name == "func.func") func_node = &node;
        if (!add_node &&
            (node.op_name == "arith.addf" || node.op_name == "arith.addi"))
            add_node = &node;
        if (func_node && add_node) break;
    }
    if (func_node) {
        MlirAnchor a;
        a.role = "function";
        a.op_name = func_node->op_name;
        a.dialect = func_node->dialect;
        a.node_id = func_node->id;
        a.location = loc_string(func_node->location);
        r.anchors.push_back(std::move(a));
    }
    if (add_node) {
        MlirAnchor a;
        a.role = "compute-op";
        a.op_name = add_node->op_name;
        a.dialect = add_node->dialect;
        a.node_id = add_node->id;
        a.location = loc_string(add_node->location);
        r.anchors.push_back(std::move(a));
    }

    // ---- Kernel stage: profile vecadd with the same N. ----
    backends::cuda::VecaddConfig vcfg;
    vcfg.num_elements = cfg.num_elements;
    vcfg.block_size = cfg.block_size;
    vcfg.repeats = cfg.repeats;
    std::string gpu_err;
    if (!backends::cuda::run_vecadd_profile(vcfg, r.gpu, &gpu_err)) {
        // Keep the compiler-stage results; report the GPU failure honestly.
        r.relation = RelationKind::CoExecuted;
        r.evidence.push_back("compiler stage complete (" +
                             std::to_string(r.mlir_ops) + " ops analyzed)");
        r.evidence.push_back("kernel stage failed: " + gpu_err);
        return fail("GPU profiling failed: " + gpu_err);
    }
    r.grid_size = r.gpu.grid_size;

    // ---- Relationship: only claim what was verified. ----
    if (embedded && has_vecadd_func && add_node && r.gpu.correct) {
        r.relation = RelationKind::SameComputation;
        r.evidence.push_back("mlir.func @vecadd analyzed" +
                             std::string(func_node ? " (node " + std::to_string(func_node->id) + ")" : ""));
        r.evidence.push_back("compute op " + add_node->op_name + " (node " +
                             std::to_string(add_node->id) + ") matches kernel arithmetic (f32 add)");
        r.evidence.push_back("profile N == mlir element count N (" +
                             std::to_string(cfg.num_elements) + ")");
        r.evidence.push_back("dtype f32 on both sides (memref<Nxf32> / float kernel)");
        r.evidence.push_back("gpu result verified correct against host reference");
    } else {
        r.relation = RelationKind::CoExecuted;
        r.evidence.push_back("mlir module analyzed (" + std::to_string(r.mlir_ops) +
                             " ops, " + std::to_string(r.mlir_funcs) + " funcs; " +
                             std::to_string(r.transform_edges) + " transform edges)");
        r.evidence.push_back("kernel " + r.kernel + " executed in the same session (N=" +
                             std::to_string(cfg.num_elements) + ", verified " +
                             std::string(r.gpu.correct ? "correct" : "INCORRECT") + ")");
        r.evidence.push_back(
            "no computation-equivalence claim: user MLIR is not the reference vecadd workload");
    }

    r.ok = true;
    return r;
#endif
}

std::string correlation_to_json(const CorrelationRecord& r) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"schema\": \"drishti.correlation/v1\",\n"
        << "  \"ok\": " << (r.ok ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(r.error) << "\",\n"
        << "  \"relation\": \"" << relation_name(r.relation) << "\",\n"
        << "  \"compiler\": {\n"
        << "    \"mlir_source\": \"" << json_escape(r.mlir_source_label) << "\",\n"
        << "    \"ops\": " << r.mlir_ops << ",\n"
        << "    \"funcs\": " << r.mlir_funcs << ",\n"
        << "    \"pipeline\": \"" << json_escape(r.pipeline) << "\",\n"
        << "    \"pipeline_pass_id\": " << r.pipeline_pass_id << ",\n"
        << "    \"transform_edges\": " << r.transform_edges << ",\n"
        << "    \"anchors\": [";
    for (std::size_t i = 0; i < r.anchors.size(); ++i) {
        const auto& a = r.anchors[i];
        oss << (i ? ", " : "") << "\n      {\"role\": \"" << json_escape(a.role)
            << "\", \"op\": \"" << json_escape(a.op_name) << "\", \"dialect\": \""
            << json_escape(a.dialect) << "\", \"node_id\": " << a.node_id
            << ", \"location\": \"" << json_escape(a.location) << "\"}";
    }
    oss << (r.anchors.empty() ? "]" : "\n    ]") << ",\n"
        << "    \"evidence\": [";
    for (std::size_t i = 0; i < r.evidence.size(); ++i) {
        oss << (i ? ", " : "") << "\n      \"" << json_escape(r.evidence[i]) << "\"";
    }
    oss << (r.evidence.empty() ? "]" : "\n    ]") << "\n  },\n";
    // Kernel stage reuses the gpu metric model JSON body (indented insert).
    oss << "  \"kernel\": {\n"
        << "    \"name\": \"" << json_escape(r.kernel) << "\",\n"
        << "    \"num_elements\": " << r.num_elements << ",\n"
        << "    \"grid_size\": " << r.grid_size << ",\n"
        << "    \"block_size\": " << r.block_size << ",\n"
        << "    \"metrics\": "
        << profiling::gpu_metrics_to_json(r.gpu) << "\n  }\n}";
    return oss.str();
}

std::string format_correlation_report(const CorrelationRecord& r) {
    std::ostringstream oss;
    oss << "Drishti Correlation Report\n"
        << "  Relation : " << relation_name(r.relation) << "\n"
        << "  MLIR     : " << r.mlir_source_label << " (" << r.mlir_ops << " ops, "
        << r.mlir_funcs << " funcs)\n";
    for (const auto& a : r.anchors) {
        oss << "    anchor : " << a.role << " " << a.op_name << " (node " << a.node_id
            << ") @" << a.location << "\n";
    }
    oss << "  Pass     : " << (r.pipeline.empty() ? "(none)" : r.pipeline)
        << " (pass id " << r.pipeline_pass_id << ", " << r.transform_edges
        << " transform edges)\n";
    if (!r.ok) {
        oss << "  Status   : FAILED: " << r.error << "\n";
        return oss.str();
    }
    oss << "  Kernel   : " << r.kernel << " (N=" << r.num_elements
        << ", grid=" << r.grid_size << ", block=" << r.block_size << ")\n"
        << "  Perf     : kernel avg " << dbl(r.gpu.kernel_ms_avg) << " ms, min "
        << dbl(r.gpu.kernel_ms_min) << " ms, " << dbl(r.gpu.gbps_effective)
        << " GB/s effective, verify " << (r.gpu.correct ? "PASS" : "FAIL") << "\n"
        << "  Chain    : ";
    bool first = true;
    for (const auto& a : r.anchors) {
        if (!first) oss << " -> ";
        oss << a.op_name << " (#" << a.node_id << ")";
        first = false;
    }
    if (!first) oss << " -> ";
    oss << "pass:" << r.pipeline << " -> kernel:" << r.kernel << " -> "
        << dbl(r.gpu.kernel_ms_avg) << " ms / " << dbl(r.gpu.gbps_effective) << " GB/s\n";
    oss << "  Evidence :\n";
    for (const auto& e : r.evidence) oss << "    - " << e << "\n";
    return oss.str();
}

drishti::provenance::PassInfo correlation_to_pass_info(const CorrelationRecord& r) {
    drishti::provenance::PassInfo p;
    p.name = "correlate:" + (r.kernel.empty() ? "unknown" : r.kernel);
    std::ostringstream oss;
    oss << relation_name(r.relation) << " N=" << r.num_elements << " anchors="
        << r.anchors.size() << " edges=" << r.transform_edges << " avg_ms="
        << dbl(r.gpu.kernel_ms_avg) << " gbps=" << dbl(r.gpu.gbps_effective);
    p.description = oss.str();
    p.enabled_by_default = true;
    return p;
}

}  // namespace drishti::correlation
