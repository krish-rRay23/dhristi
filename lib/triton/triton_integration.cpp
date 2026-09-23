#include "drishti/triton/triton_integration.h"
#include "drishti/core/config.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#if DRISHTI_HAVE_CUDA
#include "drishti/backends/cuda/cuda_backend.h"
#endif
#include "drishti/diagnosis/root_cause.h"
#include "drishti/optimizer/cost_model.h"
#include "drishti/profiling/gpu_metrics.h"
#include "drishti/provenance/provenance.h"

namespace drishti::triton {
namespace {

std::string find_python_executable() {
    // Check known Python locations with Triton installed
    const std::vector<std::string> candidates = {
        "C:\\Users\\krish\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "python.exe",
        "python",
        "python3"
    };

    for (const auto& path : candidates) {
#if defined(_WIN32)
        std::string cmd = "\"\"" + path + "\" -c \"import triton\" >nul 2>&1\"";
        int rc = std::system(cmd.c_str());
#else
        std::string cmd = "\"" + path + "\" -c \"import triton\" >/dev/null 2>&1";
        int rc = std::system(cmd.c_str());
#endif
        if (rc == 0) {
            return path;
        }
    }
    return "python";
}

std::string find_compiler_script() {
    const std::vector<std::string> candidates = {
#ifdef DRISHTI_TRITON_COMPILER_SCRIPT_PATH
        DRISHTI_TRITON_COMPILER_SCRIPT_PATH,
#endif
        "tools/triton/drishti_triton_compiler.py",
        "../tools/triton/drishti_triton_compiler.py",
        "../../tools/triton/drishti_triton_compiler.py",
        "../../../tools/triton/drishti_triton_compiler.py",
        "../../../../tools/triton/drishti_triton_compiler.py",
        "C:/Users/krish/Dhristi/tools/triton/drishti_triton_compiler.py"
    };

    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::absolute(path).string();
        }
    }
    return "tools/triton/drishti_triton_compiler.py";
}

std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            if (next == 'n') { out += '\n'; ++i; }
            else if (next == 'r') { out += '\r'; ++i; }
            else if (next == 't') { out += '\t'; ++i; }
            else if (next == '\"') { out += '\"'; ++i; }
            else if (next == '\\') { out += '\\'; ++i; }
            else { out += s[i]; }
        } else {
            out += s[i];
        }
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string extract_json_string_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\": \"";
    std::size_t pos = json.find(pattern);
    if (pos == std::string::npos) {
        pattern = "\"" + key + "\":\"";
        pos = json.find(pattern);
        if (pos == std::string::npos) return "";
    }
    pos += pattern.size();

    std::string val;
    bool in_escape = false;
    while (pos < json.size()) {
        char c = json[pos++];
        if (in_escape) {
            val += '\\';
            val += c;
            in_escape = false;
        } else if (c == '\\') {
            in_escape = true;
        } else if (c == '\"') {
            break;
        } else {
            val += c;
        }
    }
    return json_unescape(val);
}

long long extract_json_int_field(const std::string& json, const std::string& key, long long default_val = 0) {
    std::string pattern = "\"" + key + "\":";
    std::size_t pos = json.find(pattern);
    if (pos == std::string::npos) return default_val;
    pos += pattern.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    std::size_t end_pos = pos;
    while (end_pos < json.size() && (std::isdigit(json[end_pos]) || json[end_pos] == '-')) ++end_pos;
    if (end_pos == pos) return default_val;
    try {
        return std::stoll(json.substr(pos, end_pos - pos));
    } catch (...) {
        return default_val;
    }
}

bool compile_triton_kernel(
    const TritonWorkloadConfig& cfg,
    TritonCompilerArtifacts& artifacts,
    std::string* err)
{
    std::string python_bin = find_python_executable();
    std::string script_path = cfg.custom_script_path.empty() ? find_compiler_script() : cfg.custom_script_path;

    if (!std::filesystem::exists(script_path)) {
        if (err) *err = "Triton compiler script not found at: " + script_path;
        return false;
    }

#if defined(_WIN32)
    std::string cmd = "\"\"" + python_bin + "\" \"" + script_path + "\" " +
                      cfg.workload_name + " " +
                      std::to_string(cfg.block_size) + " 86\"";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    std::string cmd = "\"" + python_bin + "\" \"" + script_path + "\" " +
                      cfg.workload_name + " " +
                      std::to_string(cfg.block_size) + " 86";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif

    if (!pipe) {
        if (err) *err = "Failed to execute Python Triton compiler command: " + cmd;
        return false;
    }

    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

#if defined(_WIN32)
    int rc = _pclose(pipe);
#else
    int rc = pclose(pipe);
#endif

    if (rc != 0) {
        if (err) *err = "Python Triton compiler failed (exit code " + std::to_string(rc) + "):\n" + output;
        return false;
    }

    const std::string marker_start = "__DRISHTI_TRITON_JSON_START__";
    const std::string marker_end = "__DRISHTI_TRITON_JSON_END__";

    std::size_t s_pos = output.find(marker_start);
    std::size_t e_pos = output.find(marker_end);

    if (s_pos == std::string::npos || e_pos == std::string::npos || e_pos <= s_pos) {
        if (err) *err = "Failed to parse JSON markers from Triton compiler output:\n" + output;
        return false;
    }

    std::string json_str = output.substr(s_pos + marker_start.size(), e_pos - s_pos - marker_start.size());

    artifacts.kernel_name = extract_json_string_field(json_str, "kernel_name");
    artifacts.python_source = extract_json_string_field(json_str, "python_source");
    artifacts.ttir = extract_json_string_field(json_str, "ttir");
    artifacts.ttgir = extract_json_string_field(json_str, "ttgir");
    artifacts.llvm_ir = extract_json_string_field(json_str, "llir");
    artifacts.ptx = extract_json_string_field(json_str, "ptx");
    artifacts.target_arch = extract_json_string_field(json_str, "target_arch");
    artifacts.num_warps = static_cast<int>(extract_json_int_field(json_str, "num_warps", 4));
    artifacts.num_stages = static_cast<int>(extract_json_int_field(json_str, "num_stages", 2));
    artifacts.shared_mem_bytes = static_cast<std::size_t>(extract_json_int_field(json_str, "shared_mem_bytes", 0));
    artifacts.register_count = static_cast<int>(extract_json_int_field(json_str, "register_count", 0));

    if (artifacts.kernel_name.empty()) artifacts.kernel_name = cfg.workload_name + "_kernel";
    if (artifacts.ptx.empty()) {
        if (err) *err = "Triton compilation succeeded but PTX output was empty";
        return false;
    }

    return true;
}

}  // namespace

TritonPipelineReport run_triton_pipeline(
    const TritonWorkloadConfig& cfg,
    std::string* err)
{
    TritonPipelineReport r;
    r.workload_name = cfg.workload_name;
    r.num_elements = cfg.num_elements;
    r.block_size = cfg.block_size;
    if (cfg.workload_name == "layernorm" && cfg.block_size == 256) {
        r.block_size = 2048;
    }

    std::cerr << "[STEP 1] Starting compilation..." << std::endl;
    // 1. Compile Triton Workload
    std::string compile_err;
    if (!compile_triton_kernel(cfg, r.artifacts, &compile_err)) {
        r.ok = false;
        r.error = compile_err;
        if (err) *err = compile_err;
        return r;
    }
    std::cerr << "[STEP 2] Compilation done. Building provenance..." << std::endl;

    // Launch grid computation
    r.grid_size = static_cast<int>((cfg.num_elements + static_cast<std::size_t>(r.block_size) - 1) / static_cast<std::size_t>(r.block_size));
    int threads_per_block = r.artifacts.num_warps * 32;

    // 2. Build Provenance Graph
    provenance::ProvenanceGraph g;

    provenance::OperationNode ast_node;
    ast_node.id = 1;
    ast_node.op_name = "@" + r.artifacts.kernel_name;
    ast_node.dialect = "triton.ast";
    g.add_node(ast_node);

    provenance::OperationNode ttir_node;
    ttir_node.id = 2;
    ttir_node.op_name = "TTIR Module";
    ttir_node.dialect = "tt";
    g.add_node(ttir_node);

    provenance::OperationNode ttgir_node;
    ttgir_node.id = 3;
    ttgir_node.op_name = "TTGIR Blocked Layout";
    ttgir_node.dialect = "ttg";
    g.add_node(ttgir_node);

    provenance::OperationNode llir_node;
    llir_node.id = 4;
    llir_node.op_name = "Lowered LLVM IR";
    llir_node.dialect = "llvm";
    g.add_node(llir_node);

    provenance::OperationNode ptx_node;
    ptx_node.id = 5;
    ptx_node.op_name = "NVPTX Assembly";
    ptx_node.dialect = "nvptx";
    g.add_node(ptx_node);

    provenance::OperationNode gpu_node;
    gpu_node.id = 6;
    gpu_node.op_name = "GPU Kernel";
    gpu_node.dialect = "cuda";
    g.add_node(gpu_node);

    auto add_triton_edge = [&](std::uint64_t from, std::uint64_t to, const std::string& details) {
        provenance::TransformationEdge e;
        e.from_id = from;
        e.to_id = to;
        e.details = details;
        g.add_edge(e);
    };

    add_triton_edge(1, 2, "triton-frontend");
    add_triton_edge(2, 3, "tritongpu-convert");
    add_triton_edge(3, 4, "triton-to-llvm");
    add_triton_edge(4, 5, "nvptx-backend");
    add_triton_edge(5, 6, "cuda-driver-jit");

    r.provenance = std::move(g);

    std::cerr << "[STEP 3] Provenance done. Checking GPU execution..." << std::endl;
    // 3. GPU Execution & Verification
#if DRISHTI_HAVE_CUDA
    if (cfg.verify_gpu) {
        backends::cuda::TritonLaunchConfig launch_cfg;
        launch_cfg.ptx_text = r.artifacts.ptx;
        launch_cfg.entry_name = r.artifacts.kernel_name;
        launch_cfg.workload_name = cfg.workload_name;
        launch_cfg.num_elements = cfg.num_elements;
        launch_cfg.grid_size = r.grid_size;
        launch_cfg.block_size = threads_per_block;
        launch_cfg.num_warps = r.artifacts.num_warps;
        launch_cfg.shared_mem_bytes = r.artifacts.shared_mem_bytes;
        launch_cfg.repeats = cfg.repeats;
        launch_cfg.bytes_per_element = (cfg.workload_name == "reduction") ? 4.0 : 12.0;
        launch_cfg.flops_per_element = (cfg.workload_name == "fused_add_relu") ? 2.0 : 1.0;

        std::cerr << "[STEP 3.1] Calling run_triton_kernel..." << std::endl;
        backends::cuda::TritonRunResult run_res;
        std::string launch_err;
        if (backends::cuda::run_triton_kernel(launch_cfg, run_res, &launch_err)) {
            std::cerr << "[STEP 3.2] run_triton_kernel returned true" << std::endl;
            r.gpu_executed = true;
            r.gpu_correct = run_res.verified;
            r.gpu_metrics = run_res.metrics;
            r.measured_kernel_ms = run_res.measured_kernel_ms;
            if (r.measured_kernel_ms > 0.0) {
                double total_flops = launch_cfg.flops_per_element * static_cast<double>(cfg.num_elements);
                r.measured_tflops = (total_flops / 1e12) / (r.measured_kernel_ms / 1000.0);
            }
        } else {
            std::cerr << "[STEP 3.2] run_triton_kernel returned false: " << launch_err << std::endl;
            r.gpu_executed = false;
            r.gpu_correct = false;
            r.error = "GPU launch failed: " + launch_err;
            if (err) *err = r.error;
        }
    }
#endif

    std::cerr << "[STEP 4] Cost modeling..." << std::endl;
    // 4. White-Box Cost Modeling (Phase 13 & 15)
    optimizer::CostModelFeatures cost_feat;
    cost_feat.kernel_label = "triton:" + cfg.workload_name;
    cost_feat.num_elements = cfg.num_elements;
    cost_feat.block_size = threads_per_block;
    cost_feat.grid_size = r.grid_size;
    cost_feat.bytes_per_element = (cfg.workload_name == "reduction") ? 4.0 : 12.0;
    cost_feat.flops_per_element = (cfg.workload_name == "fused_add_relu") ? 2.0 : 1.0;
    cost_feat.launch_count = 1;
    cost_feat.regs_per_thread_est = r.artifacts.register_count > 0 ? r.artifacts.register_count : 16;
    cost_feat.shared_mem_per_block_bytes = r.artifacts.shared_mem_bytes;
    cost_feat.device = r.gpu_metrics.device;

    optimizer::HardwareCalibrationProfile calib_prof;
    r.cost_estimate = optimizer::estimate_kernel_cost_calibrated(cost_feat, calib_prof);

    if (r.gpu_executed && r.measured_kernel_ms > 0.0) {
        r.cost_validation = optimizer::validate_prediction(
            r.cost_estimate.predicted_total_ms,
            r.measured_kernel_ms);
    }

    std::cerr << "[STEP 5] Diagnosis..." << std::endl;
    // 5. Root-Cause Diagnosis (Phase 6 & 10)
    if (r.gpu_executed) {
        correlation::CorrelationRecord corr;
        corr.ok = true;
        corr.kernel = "triton:" + cfg.workload_name;
        corr.num_elements = cfg.num_elements;
        corr.block_size = threads_per_block;
        corr.grid_size = r.grid_size;
        corr.gpu = r.gpu_metrics;
        corr.relation = correlation::RelationKind::CoExecuted;

        diagnosis::DiagnosisConfig diag_cfg;
        diagnosis::KernelTraits traits;
        traits.kernel = corr.kernel;
        traits.flops_per_element = (cfg.workload_name == "fused_add_relu") ? 2.0 : 1.0;
        traits.bytes_per_element_min = (cfg.workload_name == "reduction") ? 4.0 : 12.0;
        traits.regs_per_thread_est = r.artifacts.register_count > 0 ? r.artifacts.register_count : 16;
        traits.regs_basis = "Triton PTX register allocation";
        diag_cfg.extra_traits[corr.kernel] = traits;

        r.diagnosis = diagnosis::diagnose(corr, diag_cfg);
    } else {
        r.diagnosis.ok = true;
    }
    std::cerr << "[STEP 6] Finishing run_triton_pipeline..." << std::endl;
    r.ok = cfg.verify_gpu ? (r.gpu_executed && r.gpu_correct) : true;
    return r;
}

std::string triton_pipeline_to_json(const TritonPipelineReport& r) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"ok\": " << (r.ok ? "true" : "false") << ",\n";
    if (!r.error.empty()) {
        ss << "  \"error\": \"" << json_escape(r.error) << "\",\n";
    }
    ss << "  \"workload\": \"" << json_escape(r.workload_name) << "\",\n";
    ss << "  \"num_elements\": " << r.num_elements << ",\n";
    ss << "  \"block_size\": " << r.block_size << ",\n";
    ss << "  \"grid_size\": " << r.grid_size << ",\n";

    ss << "  \"artifacts\": {\n";
    ss << "    \"kernel_name\": \"" << json_escape(r.artifacts.kernel_name) << "\",\n";
    ss << "    \"num_warps\": " << r.artifacts.num_warps << ",\n";
    ss << "    \"num_stages\": " << r.artifacts.num_stages << ",\n";
    ss << "    \"shared_mem_bytes\": " << r.artifacts.shared_mem_bytes << ",\n";
    ss << "    \"register_count\": " << r.artifacts.register_count << ",\n";
    ss << "    \"target_arch\": \"" << json_escape(r.artifacts.target_arch) << "\",\n";
    ss << "    \"ttir_size\": " << r.artifacts.ttir.size() << ",\n";
    ss << "    \"ttgir_size\": " << r.artifacts.ttgir.size() << ",\n";
    ss << "    \"llvm_ir_size\": " << r.artifacts.llvm_ir.size() << ",\n";
    ss << "    \"ptx_size\": " << r.artifacts.ptx.size() << "\n";
    ss << "  },\n";

    ss << "  \"gpu\": {\n";
    ss << "    \"executed\": " << (r.gpu_executed ? "true" : "false") << ",\n";
    ss << "    \"correct\": " << (r.gpu_correct ? "true" : "false") << ",\n";
    ss << "    \"measured_kernel_ms\": " << r.measured_kernel_ms << ",\n";
    ss << "    \"gbps_effective\": " << r.gpu_metrics.gbps_effective << ",\n";
    ss << "    \"tflops\": " << r.measured_tflops << "\n";
    ss << "  },\n";

    ss << "  \"cost_model\": {\n";
    ss << "    \"predicted_kernel_ms\": " << r.cost_estimate.predicted_kernel_ms << ",\n";
    ss << "    \"error_percent\": " << r.cost_validation.error_percent << ",\n";
    ss << "    \"is_accurate\": " << (r.cost_validation.is_accurate ? "true" : "false") << "\n";
    ss << "  }\n";
    ss << "}";
    return ss.str();
}

std::string format_triton_pipeline_report(
    const TritonPipelineReport& r,
    bool show_ir)
{
    std::ostringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "              DRISHTI PHASE 18: DEEP TRITON INTEGRATION REPORT                 \n";
    ss << "================================================================================\n";
    ss << "Workload Name       : " << r.workload_name << "\n";
    ss << "Kernel Entry        : " << r.artifacts.kernel_name << "\n";
    ss << "Vector Elements     : " << r.num_elements << "\n";
    ss << "Launch Geometry     : Grid " << r.grid_size << " blocks, Block " << (r.artifacts.num_warps * 32)
       << " threads (" << r.artifacts.num_warps << " warps)\n";
    ss << "Target Architecture : " << r.artifacts.target_arch << " (Pipeline stages: "
       << r.artifacts.num_stages << ", Shared mem: " << r.artifacts.shared_mem_bytes << " B)\n";
    ss << "Compilation Status  : " << (r.ok ? "SUCCESS (Real Triton Compiler)" : "FAILED (" + r.error + ")") << "\n";
    ss << "--------------------------------------------------------------------------------\n";

    ss << "COMPILER STAGES CAPTURED:\n";
    ss << "  1. Triton AST / Python Source : " << r.artifacts.python_source.size() << " bytes\n";
    ss << "  2. Triton-IR (TTIR dialect)   : " << r.artifacts.ttir.size() << " bytes\n";
    ss << "  3. TritonGPU-IR (TTGIR)       : " << r.artifacts.ttgir.size() << " bytes\n";
    ss << "  4. Lowered LLVM IR            : " << r.artifacts.llvm_ir.size() << " bytes\n";
    ss << "  5. Lowered NVPTX Assembly     : " << r.artifacts.ptx.size() << " bytes\n";
    ss << "--------------------------------------------------------------------------------\n";

    // Phase 19: Vectorization & Codegen Analysis
    int v4_count = 0, v2_count = 0, b32_count = 0;
    std::istringstream ptx_stream(r.artifacts.ptx);
    std::string line;
    while (std::getline(ptx_stream, line)) {
        if (line.find("ld.global.v4") != std::string::npos || line.find("st.global.v4") != std::string::npos) v4_count++;
        else if (line.find("ld.global.v2") != std::string::npos || line.find("st.global.v2") != std::string::npos) v2_count++;
        else if (line.find("ld.global.b32") != std::string::npos || line.find("st.global.b32") != std::string::npos) b32_count++;
    }

    bool is_size_4 = (r.artifacts.ttgir.find("sizePerThread = [4]") != std::string::npos);
    ss << "VECTORIZATION CODEGEN ANALYSIS (Phase 19 Case Study):\n";
    ss << "  TTGIR Layout Vector Width : " << (is_size_4 ? "4 elements/thread (sizePerThread = [4])" : "1 element/thread (sizePerThread = [1])") << "\n";
    ss << "  128-bit Vector Ops (.v4)  : " << v4_count << " instructions\n";
    ss << "  64-bit Vector Ops (.v2)   : " << v2_count << " instructions\n";
    ss << "  32-bit Scalar Ops (.b32)  : " << b32_count << " instructions\n";
    ss << "  Codegen Classification    : "
       << (v4_count > 0 ? "VECTORIZED_128BIT (16-byte alignment propagated)" : "SCALAR_DECOMPOSED (Missing 16-byte alignment hint)") << "\n";
    ss << "--------------------------------------------------------------------------------\n";

    ss << "LIVE GPU EXECUTION & VERIFICATION (CUDA Backend):\n";
    if (r.gpu_executed) {
        ss << "  Status             : " << (r.gpu_correct ? "PASS (Verified Correct)" : "FAIL (Result Mismatch)") << "\n";
        ss << "  Target GPU Device  : " << r.gpu_metrics.device.name << " (sm_"
           << r.gpu_metrics.device.compute_major << r.gpu_metrics.device.compute_minor << ")\n";
        ss << "  Kernel Latency     : " << std::fixed << std::setprecision(4) << r.measured_kernel_ms << " ms\n";
        ss << "  Effective VRAM BW  : " << std::fixed << std::setprecision(2) << r.gpu_metrics.gbps_effective << " GB/s\n";
        ss << "  Compute Throughput : " << std::fixed << std::setprecision(3) << r.measured_tflops << " TFLOPS\n";
    } else {
        ss << "  Status             : SKIPPED / NOT EXECUTED (" << r.error << ")\n";
    }
    ss << "--------------------------------------------------------------------------------\n";

    ss << "WHITE-BOX COST MODEL PREDICTION (Analytical + Calibrated):\n";
    ss << "  Predicted Latency  : " << std::fixed << std::setprecision(4) << r.cost_estimate.predicted_kernel_ms << " ms\n";
    ss << "  Measured Latency   : " << std::fixed << std::setprecision(4) << r.measured_kernel_ms << " ms\n";
    ss << "  Prediction Error   : " << std::fixed << std::setprecision(2) << r.cost_validation.error_percent
       << "% (" << (r.cost_validation.is_accurate ? "ACCURATE <= 25%" : "DEVIATION") << ")\n";
    ss << "  Limiting Regime    : " << r.cost_estimate.bottleneck_regime << " (Occupancy: "
       << std::fixed << std::setprecision(1) << r.cost_estimate.theoretical_occupancy_pct << "%)\n";
    ss << "--------------------------------------------------------------------------------\n";

    ss << "ROOT-CAUSE INTELLIGENCE & FINDINGS:\n";
    if (r.diagnosis.findings.empty()) {
        ss << "  No major performance bottlenecks detected.\n";
    } else {
        for (const auto& f : r.diagnosis.findings) {
            ss << "  - [" << f.bottleneck.rule_id << "] " << f.bottleneck.title << "\n";
            ss << "    Confidence : " << static_cast<int>(f.confidence * 100) << "%\n";
            ss << "    Explanation: " << f.explanation << "\n";
            for (const auto& ev : f.evidence) {
                ss << "      * " << ev << "\n";
            }
        }
    }
    ss << "================================================================================\n";

    if (show_ir) {
        ss << "\n--- [CAPTURED TRITON PYTHON SOURCE] ---\n" << r.artifacts.python_source << "\n";
        ss << "\n--- [CAPTURED TRITON-IR (TTIR)] ---\n" << r.artifacts.ttir << "\n";
        ss << "\n--- [CAPTURED TRITONGPU-IR (TTGIR)] ---\n" << r.artifacts.ttgir << "\n";
        ss << "\n--- [CAPTURED TRITON LOWERED LLVM IR] ---\n" << r.artifacts.llvm_ir << "\n";
        ss << "\n--- [CAPTURED LOWERED NVPTX ASSEMBLY] ---\n" << r.artifacts.ptx << "\n";
        ss << "================================================================================\n";
    }

    return ss.str();
}

provenance::PassInfo triton_pipeline_to_pass_info(const TritonPipelineReport& r) {
    provenance::PassInfo p;
    p.name = "triton-pipeline";
    p.description = "Triton Compilation -> PTX -> GPU Execution (" + r.workload_name + ")";
    return p;
}

}  // namespace drishti::triton
