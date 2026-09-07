#include "drishti/optimizer/compiler_experiment.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include "drishti/analysis/gpu_lowering.h"
#include "drishti/analysis/mlir_analysis.h"
#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/core/config.h"

namespace drishti::optimizer {
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

std::string fmt(double v, int prec = 4) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(prec);
    oss << v;
    return oss.str();
}

std::string anchor_text(const std::string& op, std::uint64_t id) {
    return op + " (node " + std::to_string(id) + ")";
}

void fill_variant_summary(CompilerVariant& v, const std::string& label,
                          const std::string& pipeline,
                          const analysis::MlirAnalysisEngine& engine,
                          const analysis::LoweredKernels& lowered,
                          double bytes_per_element) {
    v.label = label;
    v.pipeline = pipeline;
    v.mlir_ops = engine.last_stats().total_operations;
    v.mlir_funcs = engine.last_stats().function_count;
    const auto& graph = engine.provenance();
    v.pass_count = graph.passes().size();
    v.edge_count = graph.edges().size();
    std::unordered_set<std::uint64_t> erased_nodes;
    for (const auto& edge : graph.edges()) {
        if (edge.kind == provenance::TransformationKind::Erased) {
            erased_nodes.insert(edge.from_id);
        }
    }
    for (const auto& node : graph.nodes()) {
        if (erased_nodes.find(node.id) != erased_nodes.end()) continue;
        if (node.op_name == "func.func" && v.func_anchor.empty()) {
            v.func_anchor = anchor_text(node.op_name, node.id);
        } else if (node.op_name == "affine.for") {
            v.loop_anchors.push_back(anchor_text(node.op_name, node.id));
        }
    }
    v.kernels = lowered.entry_names;
    v.bytes_per_element = bytes_per_element;
}

}  // namespace

CompilerExperimentReport run_compiler_experiment(const CompilerExperimentConfig& cfg,
                                                 std::string* err) {
    CompilerExperimentReport r;
    const auto fail = [&](const std::string& msg) {
        r.ok = false;
        r.error = msg;
        if (err) *err = msg;
        return r;
    };

    if (cfg.num_elements == 0 || cfg.num_elements > (1u << 28))
        return fail("num_elements out of range (1 .. 268435456)");
    if (cfg.block_size <= 0 || cfg.block_size > 1024)
        return fail("block_size out of range (1 .. 1024)");
    if (cfg.repeats <= 0 || cfg.repeats > 1000) return fail("repeats out of range (1 .. 1000)");
    // The 2D workload maps outer dim -> blocks, inner dim -> threads, so the
    // launch geometry is compiler-fixed: grid == block == dim == sqrt(N).
    const auto dim_ll = static_cast<long long>(
        std::llround(std::sqrt(static_cast<double>(cfg.num_elements))));
    if (dim_ll <= 0 ||
        static_cast<std::size_t>(dim_ll) * static_cast<std::size_t>(dim_ll) !=
            cfg.num_elements) {
        return fail("num_elements must be a perfect square (dim x dim workload)");
    }
    const int dim = static_cast<int>(dim_ll);
    if (cfg.block_size != dim) {
        return fail("block_size must equal dim (" + std::to_string(dim) +
                    ") for the compiled mapping (outer/inner dims)");
    }

#if !DRISHTI_HAVE_CUDA
    return fail("compiler experiment requires the CUDA backend (DRISHTI_ENABLE_CUDA=OFF)");
#else
    if (!backends::cuda::device_present())
        return fail("no CUDA device/driver present");
    profiling::GpuDeviceModel dev;
    if (!backends::cuda::query_device(dev)) return fail("device query failed: " + dev.error);

    const std::string source = analysis::fusion_workload_mlir(cfg.num_elements);
    const std::string baseline_pipe = analysis::fusion_baseline_pipeline();
    const std::string fused_pipe = analysis::fusion_transformed_pipeline();

    analysis::MlirAnalysisContext ctx_base, ctx_fused;
    analysis::MlirAnalysisEngine eng_base(ctx_base), eng_fused(ctx_fused);
    std::string step_err;
    auto run_provenance = [&](analysis::MlirAnalysisEngine& eng,
                              const std::string& pipe) {
        eng.set_pass_pipeline(pipe);
#if DRISHTI_HAVE_MLIR
        auto stats = eng.analyze_string(source, &step_err);
        if (!stats) step_err = "MLIR analysis failed: " + step_err;
        return static_cast<bool>(stats);
#else
        return true;
#endif
    };
    const std::string prov_base_pipe = "canonicalize,cse";
    const std::string prov_fused_pipe = "canonicalize,affine-loop-fusion,cse";
    if (!run_provenance(eng_base, prov_base_pipe)) return fail(step_err);
    if (!run_provenance(eng_fused, prov_fused_pipe)) return fail(step_err);

    analysis::LoweredKernels low_base =
        analysis::lower_to_ptx(source, baseline_pipe, dev.compute_major,
                               dev.compute_minor, &step_err);
    if (!low_base.ok) return fail("baseline lowering failed: " + step_err);
    analysis::LoweredKernels low_fused =
        analysis::lower_to_ptx(source, fused_pipe, dev.compute_major,
                               dev.compute_minor, &step_err);
    if (!low_fused.ok) return fail("fused lowering failed: " + step_err);
    // Honesty gates: the transformation must have changed the kernel count.
    if (low_base.num_kernels() != 2) {
        return fail("baseline produced " + std::to_string(low_base.num_kernels()) +
                    " kernels, expected 2 (unfused loops)");
    }
    if (low_fused.num_kernels() != 1) {
        return fail("fused pipeline produced " + std::to_string(low_fused.num_kernels()) +
                    " kernels, expected 1 (fusion did not fire)");
    }

    fill_variant_summary(r.baseline, "baseline", baseline_pipe, eng_base, low_base, 24.0);
    fill_variant_summary(r.candidate, "fused", fused_pipe, eng_fused, low_fused, 16.0);

    auto run_variant = [&](const analysis::LoweredKernels& low, double bpe,
                           const std::string& label, CompilerVariant& v,
                           backends::cuda::PtxRunResult& pres) {
        backends::cuda::PtxLaunchConfig launch;
        launch.ptx_text = low.ptx;
        launch.kernel_label = label;
        launch.num_elements = cfg.num_elements;
        launch.grid_size = dim;
        launch.block_size = dim;
        launch.repeats = cfg.repeats;
        launch.num_buffers = 5;
        launch.output_slot = 4;
        launch.input_fills = {1.0f, 2.0f, 0.0f, 3.0f, 0.0f};
        launch.expected = 9.0f;
        launch.bytes_per_element = bpe;
        for (const auto& spec : low.launches) {
            backends::cuda::PtxKernelLaunch kl;
            kl.entry = spec.entry;
            kl.index_args = spec.index_args;
            kl.buffer_slots = spec.buffer_slots;
            launch.launches.push_back(std::move(kl));
        }
        // The compiler-described launches must cover every buffer exactly as
        // the signature requires; anything else is a hard error, never a guess.
        for (const auto& kl : launch.launches) {
            if (kl.buffer_slots.empty()) {
                step_err = "kernel " + kl.entry + " has no buffer operands";
                return false;
            }
        }
        if (!backends::cuda::run_ptx_kernel(launch, pres, &step_err)) {
            step_err = label + " execution failed: " + step_err;
            return false;
        }
        v.metrics = pres.metrics;
        v.per_kernel = pres.per_kernel;
        return true;
    };

    backends::cuda::PtxRunResult base_run, fused_run;
    if (!run_variant(low_base, 24.0, "fusedemo-base", r.baseline, base_run))
        return fail(step_err);
    if (!run_variant(low_fused, 16.0, "fusedemo-fused", r.candidate, fused_run))
        return fail(step_err);

    ExperimentResult& e = r.result;
    e.experiment_id = "compiler-fusion";
    e.candidate_id = "fuse-affine-loops";
    e.status = ExperimentStatus::Completed;
    e.baseline_desc = "unfused: 2 kernels (" + low_base.entry_names[0] + " + " +
                      low_base.entry_names[1] + ")";
    e.candidate_desc = "fused: 1 kernel (" + low_fused.entry_names[0] + ")";
    e.baseline = r.baseline.metrics;
    e.candidate = r.candidate.metrics;
    e.primary_metric = "kernel-min-ms";
    e.baseline_value = e.baseline.kernel_ms_min;
    e.candidate_value = e.candidate.kernel_ms_min;
    e.candidate_correct = e.candidate.correct;
    e.rel_improvement = e.baseline_value > 0.0
                            ? (e.baseline_value - e.candidate_value) / e.baseline_value
                            : 0.0;
    e.verdict = classify(e.baseline_value, e.candidate_value, e.baseline.ok,
                         e.candidate.ok, e.candidate_correct, 0.05, &e.verdict_reason);
    r.ok = true;
    return r;
#endif
}

std::string compiler_experiment_to_json(const CompilerExperimentReport& r) {
    std::ostringstream oss;
    auto variant_json = [&](const CompilerVariant& v) {
        std::ostringstream o;
        o << "{\"label\": \"" << json_escape(v.label) << "\", \"pipeline\": \""
          << json_escape(v.pipeline) << "\", \"mlir_ops\": " << v.mlir_ops
          << ", \"mlir_funcs\": " << v.mlir_funcs << ", \"passes\": " << v.pass_count
          << ", \"edges\": " << v.edge_count << ", \"func_anchor\": \""
          << json_escape(v.func_anchor) << "\", \"loop_anchors\": [";
        for (std::size_t i = 0; i < v.loop_anchors.size(); ++i) {
            o << (i ? ", " : "") << "\"" << json_escape(v.loop_anchors[i]) << "\"";
        }
        o << "], \"kernels\": [";
        for (std::size_t i = 0; i < v.kernels.size(); ++i) {
            o << (i ? ", " : "") << "\"" << json_escape(v.kernels[i]) << "\"";
        }
        o << "], \"bytes_per_element\": " << fmt(v.bytes_per_element, 1)
          << ", \"per_kernel_ms_min\": [";
        for (std::size_t i = 0; i < v.per_kernel.size(); ++i) {
            o << (i ? ", " : "") << "{\"entry\": \"" << json_escape(v.per_kernel[i].entry)
              << "\", \"min_ms\": " << fmt(v.per_kernel[i].ms_min) << "}";
        }
        o << "], \"metrics\": " << profiling::gpu_metrics_to_json(v.metrics) << "}";
        return o.str();
    };
    oss << "{\n"
        << "  \"schema\": \"drishti.compiler_experiment/v1\",\n"
        << "  \"ok\": " << (r.ok ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(r.error) << "\",\n"
        << "  \"baseline\": " << variant_json(r.baseline) << ",\n"
        << "  \"candidate\": " << variant_json(r.candidate) << ",\n";
    ExperimentReport shared;
    shared.ok = r.ok;
    shared.error = r.error;
    shared.relation = "same-computation";
    shared.kernel = "fusedemo";
    if (r.ok) shared.experiments.push_back(r.result);
    oss << "  \"comparison\": " << experiments_to_json(shared) << "\n}";
    return oss.str();
}

std::string format_compiler_experiment_report(const CompilerExperimentReport& r) {
    std::ostringstream oss;
    oss << "Drishti Compiler Experiment (affine-loop-fusion)\n";
    if (!r.ok) {
        oss << "  Status   : FAILED: " << r.error << "\n";
        return oss.str();
    }
    auto variant_text = [&](const CompilerVariant& v) {
        std::ostringstream o;
        o << "  [" << v.label << "] pipeline: " << v.pipeline << "\n"
          << "    mlir      : " << v.mlir_ops << " ops, " << v.mlir_funcs << " funcs, "
          << v.pass_count << " passes, " << v.edge_count << " edges\n"
          << "    anchors   : " << (v.func_anchor.empty() ? "n/a" : v.func_anchor);
        for (const auto& l : v.loop_anchors) o << ", " << l;
        o << "\n    kernels   : ";
        for (std::size_t i = 0; i < v.kernels.size(); ++i) {
            if (i) o << " + ";
            o << v.kernels[i];
        }
        o << "\n    per-kernel: ";
        for (std::size_t i = 0; i < v.per_kernel.size(); ++i) {
            if (i) o << ", ";
            o << v.per_kernel[i].entry << " min " << fmt(v.per_kernel[i].ms_min) << " ms";
        }
        o << "\n    perf      : kernel avg " << fmt(v.metrics.kernel_ms_avg)
          << " ms, min " << fmt(v.metrics.kernel_ms_min) << " ms, "
          << fmt(v.metrics.gbps_effective) << " GB/s, verify "
          << (v.metrics.correct ? "PASS" : "FAIL") << "\n";
        return o.str();
    };
    oss << variant_text(r.baseline) << variant_text(r.candidate);
    const auto& e = r.result;
    oss << "  Chain     : affine.for loops -> pass:affine-loop-fusion -> kernel:"
        << (r.candidate.kernels.empty() ? "?" : r.candidate.kernels[0]) << " -> "
        << fmt(e.candidate_value) << " ms (" << verdict_name(e.verdict) << ")\n"
        << "  Baseline  : " << e.baseline_desc << " -> " << e.primary_metric << " "
        << fmt(e.baseline_value) << " ms\n"
        << "  Candidate : " << e.candidate_desc << " -> " << e.primary_metric << " "
        << fmt(e.candidate_value) << " ms\n"
        << "  Correctness: " << (e.candidate_correct ? "PASS" : "FAIL")
        << " (host reference, both variants)\n"
        << "  Verdict   : " << verdict_name(e.verdict) << " (" << e.verdict_reason
        << ")\n"
        << "\n  Note: experiment-only; nothing was applied to source.\n";
    return oss.str();
}

drishti::provenance::PassInfo compiler_experiment_to_pass_info(
    const CompilerExperimentReport& r) {
    drishti::provenance::PassInfo p;
    p.name = "cexperiment:fusion";
    if (!r.ok) {
        p.description = "failed: " + r.error;
    } else {
        p.description = "2->1 kernels, " + verdict_name(r.result.verdict) + " (" +
                        fmt(r.result.rel_improvement * 100.0, 1) + "%)";
    }
    p.enabled_by_default = true;
    return p;
}

}  // namespace drishti::optimizer
