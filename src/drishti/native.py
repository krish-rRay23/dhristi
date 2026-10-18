"""
Native C++ Drishti Executable Discovery & Wrapper
=================================================
Manages interaction with the native C++ drishti executable, discovering
local builds, checking backend support, and streaming/parsing JSON outputs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_CACHED_BINARY_PATH: Optional[str] = None


def set_binary_path(path: str) -> None:
    """Manually override the path to the native drishti executable."""
    global _CACHED_BINARY_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Native binary not found at: {path}")
    _CACHED_BINARY_PATH = os.path.abspath(path)


def find_drishti_binary() -> Optional[str]:
    """
    Discover the native drishti binary across environment variables,
    packaged bin directories, PATH, and local build trees.
    """
    global _CACHED_BINARY_PATH
    if _CACHED_BINARY_PATH and os.path.isfile(_CACHED_BINARY_PATH):
        return _CACHED_BINARY_PATH

    # 1. Check DRISHTI_BIN environment variable
    env_bin = os.environ.get("DRISHTI_BIN")
    if env_bin and os.path.isfile(env_bin):
        _CACHED_BINARY_PATH = os.path.abspath(env_bin)
        return _CACHED_BINARY_PATH

    # 2. Check bundled package directory (e.g. src/drishti/bin or site-packages/drishti/bin)
    pkg_dir = Path(__file__).resolve().parent
    bin_names = ["drishti.exe", "drishti"] if os.name == "nt" else ["drishti"]

    for bname in bin_names:
        bundled = pkg_dir / "bin" / bname
        if bundled.is_file():
            _CACHED_BINARY_PATH = str(bundled.resolve())
            return _CACHED_BINARY_PATH

    # 3. Check PATH
    for bname in bin_names:
        which_path = shutil.which(bname)
        if which_path and os.path.isfile(which_path):
            _CACHED_BINARY_PATH = os.path.abspath(which_path)
            return _CACHED_BINARY_PATH

    # 4. Check known repository build paths (relative to package or cwd)
    repo_root = pkg_dir.parent.parent  # src/drishti -> src -> root
    search_dirs = [
        repo_root / "build-clang" / "bin",
        repo_root / "build" / "bin",
        repo_root / "build_final" / "bin",
        repo_root / "build-p17",
        repo_root / "build17" / "bin",
        repo_root / "build_ci_clean" / "bin",
        Path.cwd() / "build-clang" / "bin",
        Path.cwd() / "build" / "bin",
        Path.cwd() / "build_final" / "bin",
        Path.cwd() / "bin",
    ]

    for sdir in search_dirs:
        for bname in bin_names:
            candidate = sdir / bname
            if candidate.is_file():
                _CACHED_BINARY_PATH = str(candidate.resolve())
                return _CACHED_BINARY_PATH

    return None


def run_native_command(
    args: List[str],
    timeout: Optional[float] = 120.0,
    check: bool = False,
) -> Tuple[int, str, str]:
    """
    Execute the native drishti binary with the specified arguments.
    Returns (returncode, stdout, stderr).
    """
    bin_path = find_drishti_binary()
    if not bin_path:
        raise RuntimeError(
            "Native drishti executable not found. Build Dṛṣṭi using CMake (e.g., in build-clang/) "
            "or set the DRISHTI_BIN environment variable."
        )

    cmd = [bin_path] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    return proc.returncode, proc.stdout, proc.stderr


def run_native_json(
    args: List[str],
    timeout: Optional[float] = 120.0,
) -> Dict[str, Any]:
    """
    Execute native command with JSON output requested and parse the resulting JSON.
    """
    full_args = list(args)
    if "--json-only" not in full_args and "--json" not in full_args:
        full_args.append("--json-only")

    rc, stdout, stderr = run_native_command(full_args, timeout=timeout)
    if rc != 0:
        err_msg = stderr.strip() or stdout.strip() or f"Command failed with exit code {rc}"
        raise RuntimeError(f"Native command failed: {err_msg}")

    # Extract JSON object from stdout (stdout may contain whitespace or trailing text)
    stdout_trimmed = stdout.strip()
    s_idx = stdout_trimmed.find("{")
    e_idx = stdout_trimmed.rfind("}")
    if s_idx == -1 or e_idx == -1 or e_idx < s_idx:
        raise ValueError(f"Could not parse valid JSON from native stdout:\n{stdout}")

    json_str = stdout_trimmed[s_idx : e_idx + 1]
    return json.loads(json_str)


def get_native_info() -> Dict[str, Any]:
    """Query build and backend info from native drishti binary."""
    bin_path = find_drishti_binary()
    if not bin_path:
        return {"found": False, "path": None, "error": "Native binary not found"}

    try:
        rc, stdout, stderr = run_native_command(["--info"], timeout=10.0)
        lines = stdout.strip().splitlines()
        info_dict: Dict[str, Any] = {
            "found": True,
            "path": bin_path,
            "raw_info": stdout.strip(),
        }
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                info_dict[key.strip().lower().replace(" ", "_")] = val.strip()
        return info_dict
    except Exception as ex:
        return {"found": True, "path": bin_path, "error": str(ex)}
