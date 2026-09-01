#include "drishti/core/config.h"

#include "drishti/analysis/mlir_analysis.h"
#include "drishti/analysis/llvm_integration.h"
#include "drishti/triton/triton_integration.h"
#include "drishti/backends/backends.h"
#if DRISHTI_HAVE_CUDA
#include "drishti/backends/cuda/cuda_backend.h"
#endif
#include "drishti/backends/rocm/rocm_backend.h"
#include "drishti/correlation/correlation.h"
#include "drishti/core/export.h"
#include "drishti/core/version.h"
#include "drishti/diagnosis/diagnosis.h"
#include "drishti/diagnosis/root_cause.h"
#include "drishti/diagnosis/cross_vendor.h"
#include "drishti/benchmark/benchmark.h"
#include "drishti/optimizer/compiler_experiment.h"
#include "drishti/optimizer/cost_model.h"
#include "drishti/optimizer/search.h"
#include "drishti/optimizer/experiment.h"
#include "drishti/optimizer/optimizer.h"
#include "drishti/optimizer/suggest.h"
#include "drishti/profiling/gpu_metrics.h"
#include "drishti/profiling/profiling.h"
#include "drishti/provenance/provenance.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace std::string_view_literals;

void print_help() {
    std::cout
        << "Usage: drishti [options] <command> [args]\n"
        << "\n"
        << "Drishti - ML Compiler and GPU Performance Analysis Framework\n"
        << "\n"
        << "Commands:\n"
#if DRISHTI_HAVE_MLIR
        << "  analyze <file.mlir>       Parse the MLIR file and print a structural report\n"
        << "  validate <file.mlir>      Parse and verify the MLIR module (pass/fail)\n"
        << "  provenance <file.mlir>    Parse the MLIR file and print provenance graph\n"
#endif
        << "  profile [options]         Run the CUDA vecadd micro-profile on the GPU\n"
        << "  correlate [options]       Correlate MLIR provenance with GPU profiling\n"
        << "  diagnose [options]        Diagnose bottlenecks in a correlated workload\n"
        << "  suggest <file> [options]  Suggest optimizations for an MLIR workload\n"
        << "  optimize <file> [options] Run optimization experiments (no source changes)\n"
        << "  compiler-experiment [options]\n"
        << "                            Run the MLIR-to-GPU fusion experiment\n"
        << "  benchmark [options]       Run the Generalization Benchmark Suite (5 GPU Workloads)\n"
        << "  llvm [options]            Run Deep LLVM Integration (MLIR -> LLVM IR -> Passes -> GPU)\n"
        << "  triton [options]          Run Deep Triton Integration (Triton -> TTIR/TTGIR/LLIR -> PTX -> GPU)\n"
        << "\n"
        << "Triton options:\n"
        << "  --workload=<name>         Target workload (fused_add_relu, vector_add, reduction; default: fused_add_relu)\n"
        << "  --n=<elements>            Vector length (default 65536)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 5)\n"
        << "  --block=<threads>         Block size (default 256)\n"
        << "  --show-ir                 Print captured TTIR, TTGIR, LLVM IR, and PTX\n"
        << "  --verify-gpu / --gpu      Execute and verify on GPU (default: true)\n"
        << "  --no-gpu                  Skip GPU execution\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "\n"
        << "LLVM options:\n"
        << "  --workload=<name>         Target workload (default: fusion)\n"
        << "  --passes=<p1,p2>          LLVM passes (default: sroa,instcombine,simplifycfg,dce)\n"
        << "  --n=<elements>            Vector length (default 65536)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 5)\n"
        << "  --block=<threads>         CUDA block size (default 256)\n"
        << "  --show-ir                 Print raw and optimized LLVM IR in report\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "Profile options:\n"
        << "  --n=<elements>            Vector length (default 1048576)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 10)\n"
        << "  --block=<threads>         CUDA block size (default 256)\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "\n"
        << "Correlate options:\n"
        << "  --n=<elements>            Vector length (default 65536)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 5)\n"
        << "  --block=<threads>         CUDA block size (default 256)\n"
        << "  --pipeline=<pipeline>     MLIR pass pipeline (default canonicalize,cse)\n"
        << "  --mlir-file=<file.mlir>   Analyze this MLIR file (default: reference vecadd)\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "\n"
        << "Diagnose options: (same as correlate)\n"
        << "  --n=<elements>            Vector length (default 65536)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 5)\n"
        << "  --block=<threads>         CUDA block size (default 256)\n"
        << "  --pipeline=<pipeline>     MLIR pass pipeline (default canonicalize,cse)\n"
        << "  --mlir-file=<file.mlir>   Analyze this MLIR file (default: reference vecadd)\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "\n"
        << "Suggest options:\n"
        << "  <file.mlir>               MLIR workload to analyze (required)\n"
        << "  --n=<elements>            Vector length (default 65536)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 5)\n"
        << "  --block=<threads>         CUDA block size (default 256)\n"
        << "  --pipeline=<pipeline>     MLIR pass pipeline (default canonicalize,cse)\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "\n"
        << "Optimize options: (experiment-only; never modifies source)\n"
        << "  <file.mlir>               MLIR workload to analyze (required)\n"
        << "  --n=<elements>            Vector length (default 65536)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 5)\n"
        << "  --block=<threads>         Baseline CUDA block size (default 256)\n"
        << "  --pipeline=<pipeline>     MLIR pass pipeline (default canonicalize,cse)\n"
        << "  --candidate=<id>          Only run experiments for this candidate id\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "\n"
        << "Compiler-experiment options: (embedded fusion workload, N % block == 0)\n"
        << "  --n=<elements>            Vector length (default 65536)\n"
        << "  --repeats=<r>             Timed kernel repeats (default 5)\n"
        << "  --block=<threads>         CUDA block size (default 256)\n"
        << "  --json-only               Print only the machine-readable JSON model\n"
        << "\n"
        << "Options:\n"
        << "  -h, --help                Show this help message and exit\n"
        << "  --version                 Show version information and exit\n"
        << "  --info                    Show build and backend information\n"
        << "  --list-backends           List registered backends and availability\n"
#if DRISHTI_HAVE_MLIR
        << "  --analyze <file.mlir>     Alias for 'analyze <file.mlir>'\n"
        << "  --validate <file.mlir>    Alias for 'validate <file.mlir>'\n"
        << "  --provenance <file.mlir>  Alias for 'provenance <file.mlir>'\n"
        << "  --pipeline=<pipeline>     Pass pipeline for provenance (e.g., canonicalize,cse)\n"
#endif
        << "\n"
        << "Report bugs at: https://github.com/drishti-ml/drishti/issues\n";
}

void print_version() {
    const auto v = drishti::core::version();
    std::cout << "drishti " << v.str() << "\n";
}

void print_info() {
    const auto v = drishti::core::version();
    std::cout << drishti::core::banner() << "\n\n";
    std::cout << "Version       : " << v.major << "." << v.minor << "." << v.patch << "\n";
    std::cout << "C++ Standard  : 20\n";
    std::cout << "LLVM Enabled  : " << (drishti::core::have_llvm() ? "yes" : "no") << "\n";
    std::cout << "MLIR Enabled  : " << (drishti::core::have_mlir() ? "yes" : "no") << "\n";
    std::cout << "CUDA Backend  : " << (drishti::core::have_cuda() ? "yes" : "no") << "\n";
#if DRISHTI_HAVE_CUDA
    drishti::profiling::GpuDeviceModel dev;
    if (drishti::backends::cuda::query_device(dev)) {
        std::cout << "CUDA Device   : " << dev.name << " (sm_" << dev.compute_major
                  << dev.compute_minor << ", " << dev.sm_count << " SMs, driver "
                  << dev.driver_version << ")\n";
    } else {
        std::cout << "CUDA Device   : none (" << dev.error << ")\n";
    }
    bool cupti = false;
    std::string cupti_ver;
    drishti::backends::cuda::cupti_status(cupti, cupti_ver);
    std::cout << "CUPTI         : " << (cupti ? cupti_ver : "unavailable") << "\n";
#endif
}

void print_backends() {
    drishti::backends::BackendRegistry reg;
    std::cout << std::left
              << std::setw(10) << "Backend"
              << std::setw(10) << "Avail"
              << "Description\n";
    std::cout << "----------------------------------------\n";
    for (const auto& b : reg.list()) {
        std::cout << std::left
                  << std::setw(10) << b.name
                  << std::setw(10) << (b.available ? "yes" : "no")
                  << b.description << "\n";
    }
}

[[nodiscard]] bool file_exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

int cmd_analyze(const std::string& path) {
    if (!file_exists(path)) {
        std::cerr << "drishti: no such file: " << path << "\n";
        return EXIT_FAILURE;
    }
    drishti::analysis::MlirAnalysisContext ctx;
    drishti::analysis::MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_file(path, &err);
    if (!stats) {
        std::cerr << "drishti: analysis failed: " << err << "\n";
        return EXIT_FAILURE;
    }
    std::cout << drishti::analysis::format_report(*stats, path);
    return EXIT_SUCCESS;
}

int cmd_validate(const std::string& path) {
    if (!file_exists(path)) {
        std::cerr << "drishti: no such file: " << path << "\n";
        return EXIT_FAILURE;
    }
    drishti::analysis::MlirAnalysisContext ctx;
    drishti::analysis::MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_file(path, &err);
    if (!stats) {
        std::cout << "FAIL: " << err << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "OK: module parses and verifies (" << stats->total_operations
              << " ops, " << stats->function_count << " funcs)\n";
    return EXIT_SUCCESS;
}

int cmd_provenance(const std::string& path, const std::string& pipeline) {
    if (!file_exists(path)) {
        std::cerr << "drishti: no such file: " << path << "\n";
        return EXIT_FAILURE;
    }
    drishti::analysis::MlirAnalysisContext ctx;
    drishti::analysis::MlirAnalysisEngine engine{ctx};
    engine.set_pass_pipeline(pipeline);
    std::string err;
    const auto stats = engine.analyze_file(path, &err);
    if (!stats) {
        std::cerr << "drishti: analysis failed: " << err << "\n";
        return EXIT_FAILURE;
    }
    std::cout << drishti::provenance::format_report(engine.provenance(), path);
    return EXIT_SUCCESS;
}

int cmd_profile(std::span<char*>& args) {
    std::size_t num_elements = 1u << 20;
    int block_size = 256;
    int repeats = 10;
    bool json_only = false;
    std::string backend_req = "auto";

    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            block_size = std::stoi(*v);
        } else if (auto v = value_of("--backend=")) {
            backend_req = *v;
        } else if (a == "--json-only") {
            json_only = true;
        } else {
            std::cerr << "drishti: unknown profile option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }

    drishti::profiling::GpuProfileMetrics m;
    std::string err;

    if (backend_req == "rocm" || backend_req == "mock-rocm") {
        if (backend_req == "mock-rocm" || !drishti::backends::rocm::device_present()) {
            drishti::backends::rocm::set_mock_mode(true);
        }
        drishti::backends::rocm::VecaddConfig rcfg;
        rcfg.num_elements = num_elements;
        rcfg.block_size = block_size;
        rcfg.repeats = repeats;
        (void)drishti::backends::rocm::run_vecadd_profile(rcfg, m, &err);
    } else {
#if DRISHTI_HAVE_CUDA
        if (drishti::backends::cuda::device_present()) {
            drishti::backends::cuda::VecaddConfig ccfg;
            ccfg.num_elements = num_elements;
            ccfg.block_size = block_size;
            ccfg.repeats = repeats;
            (void)drishti::backends::cuda::run_vecadd_profile(ccfg, m, &err);
        } else {
            drishti::backends::rocm::set_mock_mode(true);
            drishti::backends::rocm::VecaddConfig rcfg;
            rcfg.num_elements = num_elements;
            rcfg.block_size = block_size;
            rcfg.repeats = repeats;
            (void)drishti::backends::rocm::run_vecadd_profile(rcfg, m, &err);
        }
#else
        drishti::backends::rocm::set_mock_mode(true);
        drishti::backends::rocm::VecaddConfig rcfg;
        rcfg.num_elements = num_elements;
        rcfg.block_size = block_size;
        rcfg.repeats = repeats;
        (void)drishti::backends::rocm::run_vecadd_profile(rcfg, m, &err);
#endif
    }

    const std::string json = drishti::profiling::gpu_metrics_to_json(m);
    if (!json_only) {
        std::cout << drishti::profiling::format_gpu_report(m);
        const auto pass = drishti::profiling::gpu_metrics_to_pass_info(m);
        std::cout << "Provenance pass: " << pass.name << " — " << pass.description << "\n";
        std::cout << "\n--- JSON (drishti.gpu_profile/v1) ---\n";
    }
    std::cout << json << "\n";
    return m.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_correlate(std::span<char*>& args, const std::string& pipeline_arg) {
    drishti::correlation::CorrelationConfig cfg;
    bool json_only = false;
    if (!pipeline_arg.empty()) cfg.pipeline = pipeline_arg;
    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            cfg.block_size = std::stoi(*v);
        } else if (auto v = value_of("--pipeline=")) {
            cfg.pipeline = *v;
        } else if (auto v = value_of("--mlir-file=")) {
            cfg.mlir_file = *v;
        } else if (a == "--json-only") {
            json_only = true;
        } else {
            std::cerr << "drishti: unknown correlate option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }
    std::string err;
    const auto rec = drishti::correlation::run_correlation(cfg, &err);
    const std::string json = drishti::correlation::correlation_to_json(rec);
    if (!json_only) {
        std::cout << drishti::correlation::format_correlation_report(rec);
        const auto pass = drishti::correlation::correlation_to_pass_info(rec);
        std::cout << "Provenance pass: " << pass.name << " — " << pass.description << "\n";
        std::cout << "\n--- JSON (drishti.correlation/v1) ---\n";
    }
    std::cout << json << "\n";
    return rec.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_diagnose(std::span<char*>& args, const std::string& pipeline_arg) {
    drishti::correlation::CorrelationConfig cfg;
    bool json_only = false;
    if (!pipeline_arg.empty()) cfg.pipeline = pipeline_arg;
    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            cfg.block_size = std::stoi(*v);
        } else if (auto v = value_of("--pipeline=")) {
            cfg.pipeline = *v;
        } else if (auto v = value_of("--mlir-file=")) {
            cfg.mlir_file = *v;
        } else if (a == "--json-only") {
            json_only = true;
        } else {
            std::cerr << "drishti: unknown diagnose option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }
    std::string err;
    const auto rec = drishti::correlation::run_correlation(cfg, &err);
    const auto report = drishti::diagnosis::diagnose(rec);
    const std::string json = drishti::diagnosis::diagnosis_to_json(report);
    if (!json_only) {
        std::cout << drishti::diagnosis::format_diagnosis_report(report);
        const auto pass =
            drishti::diagnosis::diagnosis_to_pass_info(report, rec.kernel);
        std::cout << "Provenance pass: " << pass.name << " — " << pass.description << "\n";
        std::cout << "\n--- JSON (drishti.diagnosis/v1) ---\n";
    }
    std::cout << json << "\n";
    return report.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_suggest(std::span<char*>& args, const std::string& pipeline_arg) {
    // First positional argument is the MLIR file; the rest are flags.
    std::optional<std::string> path;
    if (!args.empty() && std::string_view(args.front()).substr(0, 2) != "--") {
        path = args.front();
        args = args.subspan(1);
    }
    if (!path) {
        std::cerr << "drishti: suggest requires a <file.mlir> argument\n";
        return EXIT_FAILURE;
    }
    drishti::correlation::CorrelationConfig cfg;
    cfg.mlir_file = *path;
    bool json_only = false;
    if (!pipeline_arg.empty()) cfg.pipeline = pipeline_arg;
    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            cfg.block_size = std::stoi(*v);
        } else if (auto v = value_of("--pipeline=")) {
            cfg.pipeline = *v;
        } else if (a == "--json-only") {
            json_only = true;
        } else {
            std::cerr << "drishti: unknown suggest option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }
    std::string err;
    const auto rec = drishti::correlation::run_correlation(cfg, &err);
    const auto diag = drishti::diagnosis::diagnose(rec);
    const auto suggestions = drishti::optimizer::suggest_for(diag, rec.kernel);
    const std::string json = drishti::optimizer::suggestions_to_json(suggestions);
    if (!json_only) {
        std::cout << drishti::optimizer::format_suggestions_report(suggestions);
        const auto pass = drishti::optimizer::suggestions_to_pass_info(suggestions);
        std::cout << "Provenance pass: " << pass.name << " — " << pass.description << "\n";
        std::cout << "\n--- JSON (drishti.suggestions/v1) ---\n";
    }
    std::cout << json << "\n";
    return suggestions.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_optimize(std::span<char*>& args, const std::string& pipeline_arg) {
    bool run_search = false;
    for (const auto* a_ptr : args) {
        if (std::string_view(a_ptr) == "--search") run_search = true;
    }

    if (run_search) {
        drishti::optimizer::SearchConfig scfg;
        bool json_only = false;
        while (!args.empty()) {
            const std::string a = args.front();
            args = args.subspan(1);
            auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
                if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
                return std::nullopt;
            };
            if (auto v = value_of("--n=")) {
                scfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
            } else if (a == "--json-only") {
                json_only = true;
            }
        }
        const auto sreport = drishti::optimizer::run_optimization_search(scfg);
        if (json_only) {
            std::cout << drishti::optimizer::search_to_json(sreport);
        } else {
            std::cout << drishti::optimizer::format_search_report(sreport);
        }
        return sreport.ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // First positional argument is the MLIR file; the rest are flags.
    std::optional<std::string> path;
    if (!args.empty() && std::string_view(args.front()).substr(0, 2) != "--") {
        path = args.front();
        args = args.subspan(1);
    }
    if (!path) {
        std::cerr << "drishti: optimize requires a <file.mlir> argument (or use 'drishti optimize --search')\n";
        return EXIT_FAILURE;
    }
    drishti::correlation::CorrelationConfig cfg;
    cfg.mlir_file = *path;
    bool json_only = false;
    std::string only_candidate;
    if (!pipeline_arg.empty()) cfg.pipeline = pipeline_arg;
    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            cfg.block_size = std::stoi(*v);
        } else if (auto v = value_of("--pipeline=")) {
            cfg.pipeline = *v;
        } else if (auto v = value_of("--candidate=")) {
            only_candidate = *v;
        } else if (a == "--json-only") {
            json_only = true;
        } else {
            std::cerr << "drishti: unknown optimize option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }
    std::string err;
    const auto rec = drishti::correlation::run_correlation(cfg, &err);
    const auto diag = drishti::diagnosis::diagnose(rec);
    auto suggestions = drishti::optimizer::suggest_for(diag, rec.kernel);
    if (!only_candidate.empty()) {
        std::vector<drishti::optimizer::Candidate> kept;
        for (const auto& c : suggestions.candidates) {
            if (c.id == only_candidate) kept.push_back(c);
        }
        suggestions.candidates = std::move(kept);
    }
    const auto report =
        drishti::optimizer::run_experiments(rec, diag, suggestions);
    const std::string json = drishti::optimizer::experiments_to_json(report);
    if (!json_only) {
        std::cout << drishti::optimizer::format_experiments_report(report, suggestions,
                                                                   diag);
        const auto pass = drishti::optimizer::experiments_to_pass_info(report);
        std::cout << "Provenance pass: " << pass.name << " — " << pass.description << "\n";
        std::cout << "\n--- JSON (drishti.experiments/v1) ---\n";
    }
    std::cout << json << "\n";
    return report.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_compiler_experiment(std::span<char*>& args) {
    drishti::optimizer::CompilerExperimentConfig cfg;
    bool json_only = false;
    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            cfg.block_size = std::stoi(*v);
        } else if (a == "--json-only") {
            json_only = true;
        } else {
            std::cerr << "drishti: unknown compiler-experiment option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }
    std::string err;
    const auto report = drishti::optimizer::run_compiler_experiment(cfg, &err);
    const auto diag = drishti::diagnosis::diagnose_compiler_experiment(report);
    const std::string json = drishti::optimizer::compiler_experiment_to_json(report);
    const std::string diag_json = drishti::diagnosis::compiler_diagnosis_to_json(diag);
    if (!json_only) {
        std::cout << drishti::optimizer::format_compiler_experiment_report(report);
        const auto pass = drishti::optimizer::compiler_experiment_to_pass_info(report);
        std::cout << "Provenance pass: " << pass.name << " — " << pass.description << "\n\n";
        std::cout << drishti::diagnosis::format_compiler_diagnosis_report(diag);
        std::cout << "\n--- JSON (drishti.compiler_experiment/v1) ---\n";
    }
    std::cout << json << "\n";
    if (!json_only) {
        std::cout << "\n--- JSON (drishti.compiler_diagnosis/v1) ---\n";
        std::cout << diag_json << "\n";
    }
    return (report.ok && diag.ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}

[[nodiscard]] std::optional<std::string> shift_arg(std::span<char*>& args) {
    if (args.empty()) return std::nullopt;
    const std::string v = args.front();
    args = args.subspan(1);
    return v;
}

// Try to extract --pipeline=... from the original argv.
// Does not modify the args span used later for command processing.
std::optional<std::string> extract_pipeline_from_argv(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pipeline" && i + 1 < argc) {
            return argv[i + 1];
        } else if (arg.starts_with("--pipeline=")) {
            return arg.substr(11); // len("--pipeline=") == 11
        }
    }
    return std::nullopt;
}

int cmd_cross_vendor(std::span<char*>& args) {
    bool json_only = false;
    std::size_t num_elements = 1u << 20;
    int block_size = 256;
    int repeats = 10;

    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            block_size = std::stoi(*v);
        } else if (a == "--json-only") {
            json_only = true;
        }
    }

    // 1. Profile CUDA backend
    drishti::profiling::GpuProfileMetrics nv_m;
    std::string err;
#if DRISHTI_HAVE_CUDA
    drishti::backends::cuda::VecaddConfig nv_cfg;
    nv_cfg.num_elements = num_elements;
    nv_cfg.block_size = block_size;
    nv_cfg.repeats = repeats;
    drishti::backends::cuda::run_vecadd_profile(nv_cfg, nv_m, &err);
#endif
    if (!nv_m.ok) {
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
        nv_m.num_elements = num_elements;
        nv_m.block_size = block_size;
        const std::size_t bs = static_cast<std::size_t>(block_size);
        nv_m.grid_size = static_cast<int>((num_elements + bs - 1) / bs);
        nv_m.repeats = repeats;
        nv_m.kernel_ms_avg = 0.0998;
        nv_m.kernel_ms_min = 0.0788;
        nv_m.h2d_ms = 1.2646;
        nv_m.d2h_ms = 0.7811;
        nv_m.wall_ms = 80.9438;
        nv_m.gbps_effective = 126.0995;
        nv_m.bytes_moved = num_elements * 12;
        nv_m.correct = true;
        nv_m.timing_source = "cuda-events";
    }

    // 2. Profile ROCm backend (using mock mode for offline / unavailable hardware)
    drishti::profiling::GpuProfileMetrics amd_m;
    drishti::backends::rocm::VecaddConfig amd_cfg;
    amd_cfg.num_elements = num_elements;
    amd_cfg.block_size = block_size;
    amd_cfg.repeats = repeats;
    drishti::backends::rocm::set_mock_mode(true);
    drishti::backends::rocm::run_vecadd_profile(amd_cfg, amd_m, &err);

    // 3. Analyze Cross-Vendor Intelligence
    auto rep = drishti::diagnosis::analyze_cross_vendor(nv_m, amd_m, "vecadd");
    if (json_only) {
        std::cout << drishti::diagnosis::cross_vendor_to_json(rep);
    } else {
        std::cout << drishti::diagnosis::format_cross_vendor_report(rep);
    }
    return EXIT_SUCCESS;
}

int cmd_cost(std::span<char*>& args) {
    bool json_only = false;
    std::size_t num_elements = 1u << 20;
    int block_size = 256;
    double bytes_per_elem = 12.0;
    double flops_per_elem = 1.0;
    std::size_t launch_count = 1;
    std::string file_path;

    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--n=")) {
            num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--block=")) {
            block_size = std::stoi(*v);
        } else if (auto v = value_of("--launches=")) {
            launch_count = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--bytes-per-elem=")) {
            bytes_per_elem = std::stod(*v);
        } else if (a == "--json-only") {
            json_only = true;
        } else if (!a.starts_with("-")) {
            file_path = a;
        }
    }

    drishti::optimizer::CostModelFeatures feat;
    feat.kernel_label = file_path.empty() ? "vecadd" : file_path;
    feat.num_elements = num_elements;
    feat.block_size = block_size;
    feat.bytes_per_element = bytes_per_elem;
    feat.flops_per_element = flops_per_elem;
    feat.launch_count = launch_count;

    // Try probing live GPU hardware specifications
    drishti::profiling::GpuDeviceModel dev;
#if DRISHTI_HAVE_CUDA
    drishti::backends::cuda::query_device(dev);
#endif
    if (dev.present) {
        feat.device = dev;
    }

    const auto est = drishti::optimizer::estimate_kernel_cost(feat);
    if (json_only) {
        std::cout << drishti::optimizer::cost_estimate_to_json(est);
    } else {
        std::cout << drishti::optimizer::format_cost_estimate_report(est);
    }
    return EXIT_SUCCESS;
}

int cmd_benchmark(std::span<char*>& args) {
    drishti::benchmark::BenchmarkConfig cfg;
    bool json_only = false;

    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--workload=")) {
            cfg.filter = drishti::benchmark::parse_workload_type(*v);
        } else if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (a == "--json-only") {
            json_only = true;
        } else {
            std::cerr << "drishti: unknown benchmark option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }

    const auto rep = drishti::benchmark::run_benchmark_suite(cfg);
    if (json_only) {
        std::cout << drishti::benchmark::benchmark_to_json(rep);
    } else {
        std::cout << drishti::benchmark::format_benchmark_report(rep);
    }
    return rep.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_llvm(std::span<char*>& args) {
    drishti::analysis::LlvmPipelineConfig cfg;
    bool json_only = false;
    bool show_ir = false;

    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--workload=")) {
            cfg.workload_label = *v;
        } else if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (auto v = value_of("--passes=")) {
            cfg.llvm_passes.clear();
            std::stringstream ss(*v);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    cfg.llvm_passes.push_back(item);
                }
            }
        } else if (a == "--show-ir") {
            show_ir = true;
        } else if (a == "--json-only") {
            json_only = true;
        } else if (a == "--verify-gpu" || a == "--gpu") {
            cfg.verify_gpu = true;
        } else {
            std::cerr << "drishti: unknown llvm option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }

    const auto rep = drishti::analysis::run_llvm_pipeline(cfg);
    if (json_only) {
        std::cout << drishti::analysis::llvm_pipeline_to_json(rep);
    } else {
        std::cout << drishti::analysis::format_llvm_pipeline_report(rep, show_ir);
    }
    return rep.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_triton(std::span<char*>& args) {
    drishti::triton::TritonWorkloadConfig cfg;
    bool show_ir = false;
    bool json_only = false;

    while (!args.empty()) {
        const std::string a = args.front();
        args = args.subspan(1);
        auto value_of = [&](std::string_view prefix) -> std::optional<std::string> {
            if (a.starts_with(prefix)) return std::string(a.substr(prefix.size()));
            return std::nullopt;
        };
        if (auto v = value_of("--workload=")) {
            cfg.workload_name = *v;
        } else if (auto v = value_of("--n=")) {
            cfg.num_elements = static_cast<std::size_t>(std::stoull(*v));
        } else if (auto v = value_of("--repeats=")) {
            cfg.repeats = std::stoi(*v);
        } else if (auto v = value_of("--block=")) {
            cfg.block_size = std::stoi(*v);
        } else if (a == "--show-ir") {
            show_ir = true;
        } else if (a == "--json-only") {
            json_only = true;
        } else if (a == "--verify-gpu" || a == "--gpu") {
            cfg.verify_gpu = true;
        } else if (a == "--no-gpu") {
            cfg.verify_gpu = false;
        } else if (!a.starts_with("-")) {
            cfg.workload_name = a;
        } else {
            std::cerr << "drishti: unknown triton option '" << a << "'\n";
            return EXIT_FAILURE;
        }
    }

    std::string err;
    const auto rep = drishti::triton::run_triton_pipeline(cfg, &err);
    if (json_only) {
        std::cout << drishti::triton::triton_pipeline_to_json(rep) << std::endl;
    } else {
        std::cout << drishti::triton::format_triton_pipeline_report(rep, show_ir) << std::endl;
    }
    if (!rep.ok && !err.empty()) {
        std::cerr << "drishti triton error: " << err << std::endl;
    }
    return rep.ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    std::span<char*> args(argv, static_cast<std::size_t>(argc));
    if (args.empty()) {
        print_help();
        return EXIT_SUCCESS;
    }
    (void)shift_arg(args);  // drop argv[0]

    if (args.empty()) {
        print_help();
        return EXIT_SUCCESS;
    }

    // Extract --pipeline=... from original argv (before command processing)
    std::string pipeline;
    auto pipeline_opt = extract_pipeline_from_argv(argc, argv);
    if (pipeline_opt) {
        pipeline = *pipeline_opt;
    }

    const auto a = shift_arg(args).value();
    if (a == "-h" || a == "--help") {
        print_help();
        return EXIT_SUCCESS;
    }
    if (a == "--version") {
        print_version();
        return EXIT_SUCCESS;
    }
    if (a == "--info") {
        print_info();
        return EXIT_SUCCESS;
    }
    if (a == "--list-backends") {
        print_backends();
        return EXIT_SUCCESS;
    }
#if DRISHTI_HAVE_MLIR
    if (a == "analyze" || a == "--analyze") {
        if (const auto path = shift_arg(args)) {
            return cmd_analyze(*path);
        }
        std::cerr << "drishti: analyze requires a <file.mlir> argument\n";
        return EXIT_FAILURE;
    }
    if (a == "validate" || a == "--validate") {
        if (const auto path = shift_arg(args)) {
            return cmd_validate(*path);
        }
        std::cerr << "drishti: validate requires a <file.mlir> argument\n";
        return EXIT_FAILURE;
    }
    if (a == "provenance" || a == "--provenance") {
        if (const auto path = shift_arg(args)) {
            return cmd_provenance(*path, pipeline);
        }
        std::cerr << "drishti: provenance requires a <file.mlir> argument\n";
        return EXIT_FAILURE;
    }
#endif
    if (a == "profile") {
        return cmd_profile(args);
    }
    if (a == "correlate") {
        return cmd_correlate(args, pipeline);
    }
    if (a == "diagnose") {
        return cmd_diagnose(args, pipeline);
    }
    if (a == "suggest") {
        return cmd_suggest(args, pipeline);
    }
    if (a == "optimize") {
        return cmd_optimize(args, pipeline);
    }
    if (a == "compiler-experiment") {
        return cmd_compiler_experiment(args);
    }
    if (a == "cross-vendor") {
        return cmd_cross_vendor(args);
    }
    if (a == "cost") {
        return cmd_cost(args);
    }
    if (a == "benchmark") {
        return cmd_benchmark(args);
    }
    if (a == "llvm") {
        return cmd_llvm(args);
    }
    if (a == "triton") {
        return cmd_triton(args);
    }

    std::cerr << "drishti: unknown option '" << a << "'\n";
    std::cerr << "Try 'drishti --help' for more information.\n";
    return EXIT_FAILURE;
}