"""
Dṛṣṭi System Health & Diagnostics (Doctor)
===========================================
Detects system dependencies, GPU telemetry, Triton compilation environment,
LLVM/MLIR compilers, native C++ binary, and NVIDIA Nsight Compute availability.
"""

from __future__ import annotations

import datetime
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from drishti.compiler import find_triton_python
from drishti.native import find_drishti_binary, get_native_info
from drishti.schemas import CheckItem, DoctorReport


def probe_python() -> Dict[str, Any]:
    """Inspect current Python environment."""
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "architecture": platform.architecture()[0],
        "platform": platform.platform(),
    }


def probe_triton() -> Dict[str, Any]:
    """Inspect Triton compiler installation."""
    py_bin = find_triton_python()
    triton_info: Dict[str, Any] = {
        "available": False,
        "version": None,
        "python_executable": py_bin,
        "target": "cuda",
    }

    try:
        proc = subprocess.run(
            [py_bin, "-c", "import triton; print(triton.__version__)"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if proc.returncode == 0:
            triton_info["available"] = True
            triton_info["version"] = proc.stdout.strip()
    except Exception as ex:
        triton_info["error"] = str(ex)

    return triton_info


def probe_llvm_mlir(native_info: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect LLVM and MLIR toolchain."""
    info: Dict[str, Any] = {
        "llvm_config_version": None,
        "mlir_opt_version": None,
        "native_llvm_enabled": native_info.get("llvm_enabled", "no") == "yes",
        "native_mlir_enabled": native_info.get("mlir_enabled", "no") == "yes",
    }

    llvm_config = shutil.which("llvm-config") or shutil.which("llvm-config.exe")
    if llvm_config:
        try:
            p = subprocess.run([llvm_config, "--version"], capture_output=True, text=True, timeout=5.0)
            if p.returncode == 0:
                info["llvm_config_version"] = p.stdout.strip()
        except Exception:
            pass

    mlir_opt = shutil.which("mlir-opt") or shutil.which("mlir-opt.exe")
    if mlir_opt:
        try:
            p = subprocess.run([mlir_opt, "--version"], capture_output=True, text=True, timeout=5.0)
            if p.returncode == 0:
                lines = [l.strip() for l in p.stdout.splitlines() if l.strip()]
                info["mlir_opt_version"] = lines[0] if lines else "available"
        except Exception:
            pass

    return info


def probe_cuda_and_gpu() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Query NVIDIA driver and GPU devices via nvidia-smi."""
    driver_info: Dict[str, Any] = {
        "available": False,
        "driver_version": None,
        "cuda_version": None,
    }
    gpu_info: Dict[str, Any] = {
        "present": False,
        "name": None,
        "compute_capability": None,
        "sm_count": 0,
        "total_memory_mb": 0,
        "free_memory_mb": 0,
    }

    smi_bin = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not smi_bin:
        default_smi = Path(r"C:\Windows\System32\nvidia-smi.exe")
        if default_smi.is_file():
            smi_bin = str(default_smi)

    if smi_bin:
        try:
            p = subprocess.run(
                [smi_bin, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if p.returncode == 0 and p.stdout.strip():
                driver_info["available"] = True
                gpu_info["present"] = True
                lines = p.stdout.strip().splitlines()
                parts = [x.strip() for x in lines[0].split(",")]
                if len(parts) >= 4:
                    gpu_info["name"] = parts[0]
                    gpu_info["total_memory_mb"] = int(float(parts[1]))
                    gpu_info["free_memory_mb"] = int(float(parts[2]))
                    driver_info["driver_version"] = parts[3]
        except Exception:
            pass

        # Query CUDA version
        try:
            p = subprocess.run([smi_bin], capture_output=True, text=True, timeout=5.0)
            for line in p.stdout.splitlines():
                if "CUDA Version:" in line:
                    idx = line.find("CUDA Version:")
                    cuda_ver = line[idx + len("CUDA Version:"):].split()[0]
                    driver_info["cuda_version"] = cuda_ver.replace("|", "").strip()
                    break
        except Exception:
            pass

    # Enrich GPU info using native binary query if available
    bin_path = find_drishti_binary()
    if bin_path:
        try:
            p = subprocess.run([bin_path, "--info"], capture_output=True, text=True, timeout=5.0)
            for line in p.stdout.splitlines():
                if line.startswith("CUDA Device"):
                    # e.g., CUDA Device : NVIDIA GeForce RTX 3050 Laptop GPU (sm_86, 16 SMs, driver 13.1)
                    val = line.split(":", 1)[1].strip()
                    if "sm_" in val:
                        gpu_info["present"] = True
                        sm_idx = val.find("sm_")
                        gpu_info["compute_capability"] = val[sm_idx:sm_idx+5]
                    if " SMs" in val:
                        tokens = val.split()
                        for i, t in enumerate(tokens):
                            if t == "SMs," and i > 0:
                                try:
                                    gpu_info["sm_count"] = int(tokens[i-1])
                                except Exception:
                                    pass
        except Exception:
            pass

    return driver_info, gpu_info


def probe_nsight_compute() -> Dict[str, Any]:
    """Inspect NVIDIA Nsight Compute (ncu) profiler availability."""
    info: Dict[str, Any] = {
        "available": False,
        "path": None,
        "version": None,
    }

    ncu_bin = shutil.which("ncu") or shutil.which("ncu.exe")
    if not ncu_bin:
        # Check standard installation directory
        base_dir = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute")
        if base_dir.is_dir():
            for p in sorted(base_dir.glob("**/ncu.exe"), reverse=True):
                if p.is_file():
                    ncu_bin = str(p.resolve())
                    break

    if ncu_bin:
        info["path"] = ncu_bin
        try:
            p = subprocess.run([ncu_bin, "--version"], capture_output=True, text=True, timeout=5.0)
            if p.returncode == 0:
                info["available"] = True
                for line in p.stdout.splitlines():
                    if "Version" in line:
                        info["version"] = line.strip()
                        break
                if not info["version"]:
                    info["version"] = p.stdout.strip().splitlines()[-1]
        except Exception as ex:
            info["error"] = str(ex)

    return info


def doctor() -> DoctorReport:
    """Run full system diagnostics and build DoctorReport."""
    native_info = get_native_info()
    py_info = probe_python()
    triton_info = probe_triton()
    llvm_info = probe_llvm_mlir(native_info)
    driver_info, gpu_info = probe_cuda_and_gpu()
    ncu_info = probe_nsight_compute()

    checks: List[CheckItem] = []

    # 1. Python Check
    checks.append(CheckItem(
        name="Python Environment",
        status="pass",
        message=f"Python {py_info['version']} ({py_info['architecture']})",
        details=py_info['executable'],
    ))

    # 2. Triton Check
    if triton_info["available"]:
        checks.append(CheckItem(
            name="Triton Compiler",
            status="pass",
            message=f"Triton v{triton_info['version']} (Target: {triton_info['target']})",
            details=f"Runtime: {triton_info['python_executable']}",
        ))
    else:
        checks.append(CheckItem(
            name="Triton Compiler",
            status="warn",
            message="Triton not found in current Python or candidate paths",
            details="Install triton via 'pip install triton' or 'pip install triton-windows'",
        ))

    # 3. LLVM / MLIR Check
    if llvm_info["native_llvm_enabled"] and llvm_info["native_mlir_enabled"]:
        ver = llvm_info["llvm_config_version"] or "active"
        checks.append(CheckItem(
            name="LLVM / MLIR Framework",
            status="pass",
            message=f"LLVM & MLIR enabled in native engine (LLVM {ver})",
        ))
    elif llvm_info["llvm_config_version"]:
        checks.append(CheckItem(
            name="LLVM / MLIR Framework",
            status="pass",
            message=f"LLVM {llvm_info['llvm_config_version']} detected on PATH",
        ))
    else:
        checks.append(CheckItem(
            name="LLVM / MLIR Framework",
            status="warn",
            message="LLVM/MLIR tools not found on PATH (Native engine will use built-in passes)",
        ))

    # 4. CUDA Driver Check
    if driver_info["available"]:
        checks.append(CheckItem(
            name="CUDA Driver",
            status="pass",
            message=f"NVIDIA Driver {driver_info['driver_version']} (CUDA {driver_info['cuda_version']})",
        ))
    else:
        checks.append(CheckItem(
            name="CUDA Driver",
            status="warn",
            message="nvidia-smi not detected or CUDA driver unavailable",
        ))

    # 5. GPU Hardware Check
    if gpu_info["present"]:
        sm_text = f" ({gpu_info['compute_capability']}, {gpu_info['sm_count']} SMs)" if gpu_info['compute_capability'] else ""
        vram_text = f" [{gpu_info['total_memory_mb']} MB VRAM]" if gpu_info['total_memory_mb'] else ""
        checks.append(CheckItem(
            name="GPU Accelerator",
            status="pass",
            message=f"{gpu_info['name']}{sm_text}{vram_text}",
        ))
    else:
        checks.append(CheckItem(
            name="GPU Accelerator",
            status="warn",
            message="No active NVIDIA GPU detected; mock execution mode available",
        ))

    # 6. Native Drishti Binary Check
    bin_path = find_drishti_binary()
    if bin_path:
        checks.append(CheckItem(
            name="Native Engine Binary",
            status="pass",
            message=f"Native binary located ({native_info.get('version', 'v0.1.0')})",
            details=bin_path,
        ))
    else:
        checks.append(CheckItem(
            name="Native Engine Binary",
            status="fail",
            message="drishti native executable not found. Build via CMake or set DRISHTI_BIN.",
        ))

    # 7. Nsight Compute Check (Optional)
    if ncu_info["available"]:
        checks.append(CheckItem(
            name="Nsight Compute (ncu)",
            status="pass",
            message=f"{ncu_info['version']}",
            details=ncu_info["path"],
        ))
    else:
        checks.append(CheckItem(
            name="Nsight Compute (ncu)",
            status="warn",
            message="Nsight Compute not in PATH (hardware event counters disabled; CUDA event profiling active)",
        ))

    all_pass = all(c.status != "fail" for c in checks)
    return DoctorReport(
        schema="drishti.doctor/v1",
        ok=all_pass,
        timestamp=datetime.datetime.now().isoformat(),
        python=py_info,
        triton=triton_info,
        llvm_mlir=llvm_info,
        cuda_driver=driver_info,
        gpu=gpu_info,
        native_binary=native_info,
        nsight_compute=ncu_info,
        checks=checks,
        summary="All critical systems operational." if all_pass else "Critical dependencies missing.",
    )


def format_doctor_report(report: DoctorReport) -> str:
    """Format doctor report for handsome terminal display."""
    lines = [
        "",
        "================================================================================",
        "                       DRISHTI SYSTEM HEALTH & DOCTOR REPORT                    ",
        "================================================================================",
    ]

    badge_map = {
        "pass": "[PASS]",
        "warn": "[WARN]",
        "fail": "[FAIL]",
    }

    for c in report.checks:
        tag = badge_map.get(c.status, "[INFO]")
        lines.append(f"  {tag:<7} {c.name:<24} : {c.message}")
        if c.details:
            lines.append(f"          -> Details: {c.details}")

    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"STATUS  : {'HEALTHY - Ready for GPU compilation & profiling' if report.ok else 'ATTENTION REQUIRED'}")
    lines.append(f"SUMMARY : {report.summary}")
    lines.append("================================================================================\n")
    return "\n".join(lines)
