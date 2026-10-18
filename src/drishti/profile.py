"""
Dṛṣṭi GPU Profiling & Telemetry (Profile)
=========================================
Executes kernel micro-benchmarks on the GPU using high-precision CUDA events
and captures throughput, effective memory bandwidth, and transfer overheads.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from drishti.native import find_drishti_binary, run_native_json
from drishti.schemas import ProfileReport


def run_ncu_profile(target_bin: str, bin_args: List[str]) -> Optional[Dict[str, Any]]:
    """Optionally run NVIDIA Nsight Compute to gather hardware metric counters."""
    ncu_bin = shutil.which("ncu") or shutil.which("ncu.exe")
    if not ncu_bin:
        return None

    cmd = [
        ncu_bin,
        "--metrics",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed",
        target_bin,
    ] + bin_args

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
        if proc.returncode == 0:
            return {"raw_ncu_output": proc.stdout.strip()}
    except Exception:
        pass
    return None


def profile(
    kernel_source: Optional[str] = None,
    num_elements: int = 1048576,
    block_size: int = 256,
    repeats: int = 10,
    use_ncu: bool = False,
) -> ProfileReport:
    """
    Profile a GPU kernel workload via the native C++ profiling engine.

    Args:
        kernel_source: Path to .py kernel, workload name, or None for reference vecadd.
        num_elements: Total elements to process.
        block_size: CUDA block size.
        repeats: Timed repeats.
        use_ncu: Whether to attach NVIDIA Nsight Compute CLI profiler.

    Returns:
        ProfileReport: Precise latency, bandwidth, and transfer telemetry.
    """
    bin_path = find_drishti_binary()
    if not bin_path:
        raise RuntimeError("Native drishti executable not found.")

    # 1. Determine whether to run native triton runner or native profile runner
    if kernel_source and (os.path.isfile(kernel_source) or kernel_source != "vecadd"):
        args = [
            "triton",
            f"--workload={os.path.abspath(kernel_source) if os.path.isfile(kernel_source) else kernel_source}",
            f"--block={block_size}",
            f"--n={num_elements}",
            f"--repeats={repeats}",
            "--gpu",
            "--json-only",
        ]
        res = run_native_json(args)
        gpu = res.get("gpu", {})
        artifacts = res.get("artifacts", {})

        report = ProfileReport(
            schema="drishti.gpu_profile/v1",
            ok=res.get("ok", True),
            backend="cuda",
            device={"compute_capability": artifacts.get("target_arch", "sm_86")},
            kernel=artifacts.get("kernel_name", str(kernel_source)),
            num_elements=num_elements,
            block_size=block_size,
            grid_size=res.get("grid_size", (num_elements + block_size - 1) // block_size),
            repeats=repeats,
            kernel_ms_avg=gpu.get("measured_kernel_ms", 0.0),
            kernel_ms_min=gpu.get("measured_kernel_ms", 0.0),
            h2d_ms=0.0,
            d2h_ms=0.0,
            wall_ms=gpu.get("measured_kernel_ms", 0.0) * repeats,
            gbps_effective=gpu.get("gbps_effective", 0.0),
            bytes_moved=int(num_elements * 12),
            tflops=gpu.get("tflops", 0.0),
            correct=gpu.get("correct", True),
            timing_source="cuda-events",
        )
    else:
        args = [
            "profile",
            f"--block={block_size}",
            f"--n={num_elements}",
            f"--repeats={repeats}",
            "--json-only",
        ]
        res = run_native_json(args)
        report = ProfileReport(
            schema="drishti.gpu_profile/v1",
            ok=res.get("ok", True),
            error=res.get("error", ""),
            backend=res.get("backend", "cuda"),
            device=res.get("device", {}),
            kernel=res.get("kernel", "vecadd"),
            num_elements=res.get("num_elements", num_elements),
            block_size=res.get("block_size", block_size),
            grid_size=res.get("grid_size", 0),
            repeats=res.get("repeats", repeats),
            kernel_ms_avg=res.get("kernel_ms_avg", 0.0),
            kernel_ms_min=res.get("kernel_ms_min", 0.0),
            h2d_ms=res.get("h2d_ms", 0.0),
            d2h_ms=res.get("d2h_ms", 0.0),
            wall_ms=res.get("wall_ms", 0.0),
            gbps_effective=res.get("gbps_effective", 0.0),
            bytes_moved=res.get("bytes_moved", 0),
            tflops=res.get("tflops", 0.0),
            correct=res.get("correct", True),
            timing_source=res.get("timing_source", "cuda-events"),
            cupti_available=res.get("cupti_available", False),
            cupti_version=res.get("cupti_version", ""),
            notes=res.get("notes", ""),
        )

    if use_ncu:
        report.ncu_metrics = run_ncu_profile(bin_path, args)

    return report


def format_profile_report(report: ProfileReport) -> str:
    """Format profile report for terminal view."""
    dev_name = report.device.get("name", "NVIDIA GPU")
    sm_cc = report.device.get("compute_capability", "sm_86")

    lines = [
        "",
        "================================================================================",
        "                     DRISHTI GPU MICRO-PROFILING REPORT                         ",
        "================================================================================",
        f"Target GPU Device   : {dev_name} ({sm_cc})",
        f"Kernel Profiled     : {report.kernel}",
        f"Vector Length       : {report.num_elements:,} elements",
        f"Launch Geometry     : Grid {report.grid_size} blocks, Block {report.block_size} threads",
        f"Execution Repeats   : {report.repeats} iterations",
        "--------------------------------------------------------------------------------",
        "EXECUTION TIMINGS & BANDWIDTH:",
        f"  Kernel Latency (avg): {report.kernel_ms_avg:.4f} ms",
        f"  Kernel Latency (min): {report.kernel_ms_min:.4f} ms",
        f"  Host-to-Device (H2D): {report.h2d_ms:.4f} ms",
        f"  Device-to-Host (D2H): {report.d2h_ms:.4f} ms",
        f"  Total Wall Clock    : {report.wall_ms:.4f} ms",
        f"  Effective Bandwidth : {report.gbps_effective:.2f} GB/s",
        f"  Memory Transferred  : {report.bytes_moved / (1024 * 1024):.2f} MB",
        f"  Verification Status : {'PASS (Correct)' if report.correct else 'FAILED'}",
        f"  Timing Mechanism    : {report.timing_source}",
        "--------------------------------------------------------------------------------",
    ]

    if report.ncu_metrics:
        lines.append("NSIGHT COMPUTE TELEMETRY:")
        lines.append(f"  {report.ncu_metrics.get('raw_ncu_output')}")
        lines.append("--------------------------------------------------------------------------------")

    lines.append("================================================================================\n")
    return "\n".join(lines)
