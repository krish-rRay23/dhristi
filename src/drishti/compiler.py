"""
Triton Compiler Integration & IR Extraction
============================================
Coordinates with the Triton compilation pipeline to extract multi-level
intermediate representations (AST, TTIR, TTGIR, LLVM IR, PTX) and metadata.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from drishti.schemas import CodegenMetrics, PipelineStages


def find_triton_python() -> str:
    """
    Find a Python executable that has Triton installed.
    Checks the current runtime first, then known local installation paths.
    """
    # 1. Check current Python
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import triton; print(triton.__version__)"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if proc.returncode == 0:
            return sys.executable
    except Exception:
        pass

    # 2. Check candidate Python paths on Windows
    candidates = [
        r"C:\Users\krish\AppData\Local\Programs\Python\Python311\python.exe",
        r"C:\Python311\python.exe",
        "python3.11",
        "python3",
        "python",
    ]

    for cand in candidates:
        resolved = shutil.which(cand) or cand
        if os.path.isfile(resolved):
            try:
                proc = subprocess.run(
                    [resolved, "-c", "import triton; print(triton.__version__)"],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if proc.returncode == 0:
                    return os.path.abspath(resolved)
            except Exception:
                continue

    return sys.executable


def find_compiler_script() -> str:
    """Locate the drishti_triton_compiler.py script."""
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    candidates = [
        repo_root / "tools" / "triton" / "drishti_triton_compiler.py",
        Path.cwd() / "tools" / "triton" / "drishti_triton_compiler.py",
        Path("C:/Users/krish/Dhristi/tools/triton/drishti_triton_compiler.py"),
    ]

    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())

    raise FileNotFoundError(
        "drishti_triton_compiler.py not found in tools/triton/"
    )


def compile_triton_workload(
    workload: str,
    block_size: int = 256,
    arch: int = 86,
    num_warps: int = 4,
    num_stages: int = 2,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Execute the Triton compiler to capture all lowering stages and codegen artifacts.
    workload can be a known workload name or a path to a .py kernel file.
    """
    py_bin = find_triton_python()
    script_path = find_compiler_script()

    # If workload is a relative path to an existing file, make it absolute
    if os.path.isfile(workload):
        workload = os.path.abspath(workload)

    cmd = [
        py_bin,
        script_path,
        workload,
        str(block_size),
        str(arch),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            f"Triton compilation failed (exit code {proc.returncode}):\n{err_msg}"
        )

    output = proc.stdout
    marker_start = "__DRISHTI_TRITON_JSON_START__"
    marker_end = "__DRISHTI_TRITON_JSON_END__"

    s_pos = output.find(marker_start)
    e_pos = output.find(marker_end)

    if s_pos == -1 or e_pos == -1 or e_pos <= s_pos:
        raise ValueError(
            f"Failed to parse Triton compiler JSON output markers:\n{output}"
        )

    json_str = output[s_pos + len(marker_start) : e_pos].strip()
    return json.loads(json_str)


def analyze_codegen_metrics(artifacts: Dict[str, Any]) -> CodegenMetrics:
    """Analyze vectorization and instruction decomposition from PTX and TTGIR."""
    ptx = artifacts.get("ptx", "")
    ttgir = artifacts.get("ttgir", "")

    v4_count = 0
    v2_count = 0
    b32_count = 0

    for line in ptx.splitlines():
        if "ld.global.v4" in line or "st.global.v4" in line:
            v4_count += 1
        elif "ld.global.v2" in line or "st.global.v2" in line:
            v2_count += 1
        elif "ld.global.b32" in line or "st.global.b32" in line:
            b32_count += 1

    is_size_4 = "sizePerThread = [4]" in ttgir
    vector_width = "4 elements/thread" if is_size_4 else "1 element/thread"

    if v4_count > 0:
        classification = "VECTORIZED_128BIT (16-byte alignment propagated)"
    elif v2_count > 0:
        classification = "VECTORIZED_64BIT (8-byte alignment propagated)"
    else:
        classification = "SCALAR_DECOMPOSED (Missing 16-byte alignment hint)"

    return CodegenMetrics(
        num_warps=artifacts.get("num_warps", 4),
        num_stages=artifacts.get("num_stages", 2),
        shared_mem_bytes=artifacts.get("shared_mem_bytes", 0),
        register_count=artifacts.get("register_count", 0),
        target_arch=artifacts.get("target_arch", "sm_86"),
        vector_width=vector_width,
        v4_count=v4_count,
        v2_count=v2_count,
        b32_count=b32_count,
        classification=classification,
    )
