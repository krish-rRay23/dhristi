"""
Dṛṣṭi Kernel Compiler Analysis (Analyze)
=========================================
Coordinates end-to-end multi-stage lowering analysis across:
Triton AST -> TTIR -> TTGIR -> Lowered LLVM IR -> Lowered NVPTX Assembly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from drishti.compiler import analyze_codegen_metrics, compile_triton_workload
from drishti.native import find_drishti_binary, run_native_json
from drishti.schemas import (
    AnalysisReport,
    CostModelSummary,
    GpuExecutionSummary,
    PipelineStages,
)


def analyze(
    kernel_source: str,
    block_size: int = 256,
    num_elements: int = 65536,
    arch: int = 86,
    num_warps: int = 4,
    num_stages: int = 2,
    verify_gpu: bool = True,
    repeats: int = 5,
) -> AnalysisReport:
    """
    Run compiler lowering and codegen analysis for a Triton kernel or workload.

    Args:
        kernel_source: Path to .py kernel file or registered workload name.
        block_size: GPU block / thread count.
        num_elements: Total tensor elements.
        arch: CUDA compute capability (default: 86 for sm_86).
        num_warps: Thread warps per block (default: 4).
        num_stages: Pipeline software stages (default: 2).
        verify_gpu: Whether to run and verify execution on the GPU.
        repeats: Timed repeats for execution.

    Returns:
        AnalysisReport: Complete compilation stages, codegen metrics, and telemetry.
    """
    source_file = None
    workload_name = kernel_source
    if os.path.isfile(kernel_source):
        source_file = os.path.abspath(kernel_source)
        workload_name = source_file

    # 1. Compile through Triton pipeline (capturing all IR stages)
    raw_artifacts = compile_triton_workload(
        workload=workload_name,
        block_size=block_size,
        arch=arch,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    codegen = analyze_codegen_metrics(raw_artifacts)
    stages = PipelineStages(
        python_source=raw_artifacts.get("python_source", ""),
        ttir=raw_artifacts.get("ttir", ""),
        ttgir=raw_artifacts.get("ttgir", ""),
        llvm_ir=raw_artifacts.get("llir", ""),
        ptx=raw_artifacts.get("ptx", ""),
        python_source_size=len(raw_artifacts.get("python_source", "")),
        ttir_size=len(raw_artifacts.get("ttir", "")),
        ttgir_size=len(raw_artifacts.get("ttgir", "")),
        llvm_ir_size=len(raw_artifacts.get("llir", "")),
        ptx_size=len(raw_artifacts.get("ptx", "")),
    )

    kernel_name = raw_artifacts.get("kernel_name", "triton_kernel")
    grid_size = (num_elements + block_size - 1) // block_size

    # 2. Invoke native binary if available to obtain calibrated cost model & GPU verification
    gpu_summary = GpuExecutionSummary()
    cost_summary = CostModelSummary()
    findings: List[Dict[str, Any]] = []
    recommendations: List[str] = []

    native_bin = find_drishti_binary()
    if native_bin:
        try:
            native_args = [
                "triton",
                f"--workload={workload_name}",
                f"--block={block_size}",
                f"--n={num_elements}",
                f"--repeats={repeats}",
                "--json-only",
            ]
            if not verify_gpu:
                native_args.append("--no-gpu")

            res = run_native_json(native_args, timeout=60.0)
            if "gpu" in res:
                gpu_res = res["gpu"]
                gpu_summary.executed = gpu_res.get("executed", False)
                gpu_summary.correct = gpu_res.get("correct", False)
                gpu_summary.measured_kernel_ms = gpu_res.get("measured_kernel_ms", 0.0)
                gpu_summary.gbps_effective = gpu_res.get("gbps_effective", 0.0)
                gpu_summary.tflops = gpu_res.get("tflops", 0.0)

            if "cost_model" in res:
                cm = res["cost_model"]
                cost_summary.predicted_kernel_ms = cm.get("predicted_kernel_ms", 0.0)
                cost_summary.error_percent = cm.get("error_percent", 0.0)
                cost_summary.is_accurate = cm.get("is_accurate", False)

        except Exception:
            pass

    # 3. Generate actionable compiler recommendations
    if codegen.v4_count == 0:
        recommendations.append(
            "Vectorization: PTX codegen is scalar-decomposed (.b32). Enforce 16-byte alignment hints using "
            "'tl.max_contiguous(tl.multiple_of(ptr, 16), 16)' to enable 128-bit (.v4) memory instructions."
        )
    else:
        recommendations.append(
            f"Vectorization: Efficient 128-bit vectorization confirmed ({codegen.v4_count} .v4 instructions)."
        )

    if block_size < 128:
        recommendations.append(
            f"Block Geometry: Current block size ({block_size}) may under-utilize SM warp schedulers. "
            "Consider testing 128 or 256 threads per block."
        )

    if gpu_summary.gbps_effective > 0.0 and gpu_summary.gbps_effective < 30.0:
        recommendations.append(
            f"Memory Bandwidth: Measured throughput ({gpu_summary.gbps_effective:.2f} GB/s) is transfer-bound. "
            "Batch consecutive elementwise kernels or fuse dependent stages."
        )

    return AnalysisReport(
        schema="drishti.analyze/v1",
        ok=True,
        workload=workload_name,
        kernel_name=kernel_name,
        source_file=source_file,
        num_elements=num_elements,
        block_size=block_size,
        grid_size=grid_size,
        stages=stages,
        codegen=codegen,
        gpu=gpu_summary,
        cost_model=cost_summary,
        findings=findings,
        recommendations=recommendations,
    )


def format_analysis_report(report: AnalysisReport, show_ir: bool = False) -> str:
    """Format the AnalysisReport for terminal presentation."""
    lines = [
        "",
        "================================================================================",
        "                   DRISHTI ML COMPILER LOWERING ANALYSIS REPORT                 ",
        "================================================================================",
        f"Workload Name       : {report.workload}",
        f"Kernel Entry        : {report.kernel_name}",
        f"Vector Elements     : {report.num_elements}",
        f"Launch Geometry     : Grid {report.grid_size} blocks, Block {report.block_size} threads ({report.codegen.num_warps} warps)",
        f"Target Architecture : {report.codegen.target_arch} (Stages: {report.codegen.num_stages}, Shared Mem: {report.codegen.shared_mem_bytes} B)",
        "--------------------------------------------------------------------------------",
        "COMPILER PIPELINE STAGES CAPTURED:",
        f"  1. Triton AST / Python Source : {report.stages.python_source_size:>6} bytes",
        f"  2. Triton-IR (TTIR dialect)   : {report.stages.ttir_size:>6} bytes",
        f"  3. TritonGPU-IR (TTGIR)       : {report.stages.ttgir_size:>6} bytes",
        f"  4. Lowered LLVM IR            : {report.stages.llvm_ir_size:>6} bytes",
        f"  5. Lowered NVPTX Assembly     : {report.stages.ptx_size:>6} bytes",
        "--------------------------------------------------------------------------------",
        "VECTORIZATION & CODEGEN METRICS:",
        f"  TTGIR Layout Vector Width : {report.codegen.vector_width}",
        f"  128-bit Vector Ops (.v4)  : {report.codegen.v4_count} instructions",
        f"  64-bit Vector Ops (.v2)   : {report.codegen.v2_count} instructions",
        f"  32-bit Scalar Ops (.b32)  : {report.codegen.b32_count} instructions",
        f"  Codegen Classification    : {report.codegen.classification}",
        "--------------------------------------------------------------------------------",
    ]

    if report.gpu.executed:
        lines.extend([
            "LIVE GPU EXECUTION & VERIFICATION:",
            f"  Status             : {'PASS (Verified Correct)' if report.gpu.correct else 'FAILED'}",
            f"  Measured Latency   : {report.gpu.measured_kernel_ms:.4f} ms",
            f"  Effective VRAM BW  : {report.gpu.gbps_effective:.2f} GB/s",
            f"  Compute Throughput : {report.gpu.tflops:.4f} TFLOPS",
            "--------------------------------------------------------------------------------",
        ])

    if report.cost_model.predicted_kernel_ms > 0.0:
        lines.extend([
            "WHITE-BOX COST MODEL PREDICTION:",
            f"  Predicted Latency  : {report.cost_model.predicted_kernel_ms:.4f} ms",
            f"  Measured Latency   : {report.gpu.measured_kernel_ms:.4f} ms",
            f"  Prediction Error   : {report.cost_model.error_percent:.2f}% ({'ACCURATE' if report.cost_model.is_accurate else 'DEVIATION'})",
            "--------------------------------------------------------------------------------",
        ])

    if report.recommendations:
        lines.append("COMPILER INSIGHTS & RECOMMENDATIONS:")
        for rec in report.recommendations:
            lines.append(f"  * {rec}")
        lines.append("--------------------------------------------------------------------------------")

    if show_ir:
        lines.extend([
            "",
            "--- [1. TRITON PYTHON SOURCE] ---",
            report.stages.python_source,
            "",
            "--- [2. TRITON-IR (TTIR)] ---",
            report.stages.ttir,
            "",
            "--- [3. TRITONGPU-IR (TTGIR)] ---",
            report.stages.ttgir,
            "",
            "--- [4. LOWERED LLVM IR] ---",
            report.stages.llvm_ir,
            "",
            "--- [5. LOWERED NVPTX ASSEMBLY] ---",
            report.stages.ptx,
            "",
        ])

    lines.append("================================================================================\n")
    return "\n".join(lines)
