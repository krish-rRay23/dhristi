#!/usr/bin/env python3
"""
NVIDIA Nsight Compute CLI (NCU) Telemetry Collector for Drishti Workloads
==========================================================================
Collects real dynamic GPU utilization and warp occupancy metrics via NCU CLI
where available and permitted. Handles ERR_NVGPUCTRPERM gracefully by recording
provenance status and setting occupancy to N/A without altering benchmark data.
"""

import json
import os
import shutil
import subprocess
import sys

def find_ncu_binary():
    # 1. Check PATH
    ncu_path = shutil.which("ncu")
    if ncu_path:
        return ncu_path
    
    # 2. Check standard Windows Nsight Compute paths
    win_ncu = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\ncu.bat"
    if os.path.exists(win_ncu):
        return win_ncu
    
    # Check general Nsight Compute directory pattern
    base_dir = r"C:\Program Files\NVIDIA Corporation"
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if "Nsight Compute" in item:
                cand = os.path.join(base_dir, item, "ncu.bat")
                if os.path.exists(cand):
                    return cand
    return None

def collect_ncu_telemetry(workload="fused_add_relu"):
    ncu_bin = find_ncu_binary()
    drishti_bin = "build-clang/bin/drishti.exe" if os.path.exists("build-clang/bin/drishti.exe") else "build/bin/drishti.exe"

    result = {
        "workload": workload,
        "ncu_available": ncu_bin is not None,
        "ncu_binary_path": ncu_bin if ncu_bin else "N/A",
        "ncu_status": "UNAVAILABLE",
        "occupancy": "N/A",
        "gpu_utilization_pct": "N/A",
        "notes": ""
    }

    if not ncu_bin:
        result["notes"] = "NCU CLI binary (ncu / ncu.bat) not found on system PATH."
        return result

    if not os.path.exists(drishti_bin):
        result["notes"] = f"Drishti executable binary ({drishti_bin}) not found."
        return result

    print(f"[NCU Telemetry] Found NCU at: {ncu_bin}")
    print(f"[NCU Telemetry] Profiling workload: {workload}...")

    metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    ]

    cmd = [
        ncu_bin,
        "--metrics", ",".join(metrics),
        drishti_bin, "triton", f"--workload={workload}", "--verify-gpu", "--json-only"
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = proc.stdout + "\n" + proc.stderr

        if "ERR_NVGPUCTRPERM" in output:
            result["ncu_status"] = "ERR_NVGPUCTRPERM"
            result["occupancy"] = "N/A (NCU ERR_NVGPUCTRPERM)"
            result["gpu_utilization_pct"] = "N/A (NCU ERR_NVGPUCTRPERM)"
            result["notes"] = (
                "ERR_NVGPUCTRPERM: NVIDIA GPU Performance Counter security restriction. "
                "Target device requires administrator privileges under WDDM or NVreg_RestrictProfilingToAdminUsers=0 on Linux."
            )
            print(f"[NCU Telemetry] Captured Security Limit: {result['notes']}")
        elif proc.returncode == 0:
            result["ncu_status"] = "PROFILED_SUCCESS"
            result["notes"] = "Successfully executed NCU metrics profile."
            # Parse output for metrics if printed
            for line in output.splitlines():
                if "sm__warps_active" in line:
                    result["occupancy"] = line.strip()
                if "sm__throughput" in line:
                    result["gpu_utilization_pct"] = line.strip()
            print(f"[NCU Telemetry] Success: {result['notes']}")
        else:
            result["ncu_status"] = "PROFILE_FAILED"
            result["notes"] = f"NCU exited with code {proc.returncode}"
    except Exception as e:
        result["ncu_status"] = "EXECUTION_ERROR"
        result["notes"] = str(e)

    return result

if __name__ == "__main__":
    res = collect_ncu_telemetry("fused_add_relu")
    print("\n--- NCU TELEMETRY REPORT ---")
    print(json.dumps(res, indent=2))
