"""
Unit & Integration Tests for Dṛṣṭi Python CLI and API
======================================================
Verifies:
  1. drishti doctor (health diagnostics)
  2. drishti analyze (multi-stage lowering AST -> TTIR -> TTGIR -> LLVM -> PTX)
  3. drishti profile (GPU metrics and bandwidth)
  4. drishti diagnose (bottleneck intelligence and findings)
  5. drishti benchmark (5-workload suite)
  6. drishti report (standalone HTML report generation)
  7. Python API imports and programmatic usage
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

import drishti
from drishti.schemas import (
    AnalysisReport,
    BenchmarkReport,
    DiagnosisReport,
    DoctorReport,
    ProfileReport,
    ReportOutput,
)


@pytest.fixture
def sample_kernel_path():
    path = Path("examples/fused_add_relu.py").resolve()
    assert path.is_file(), f"Sample kernel not found at {path}"
    return str(path)


def test_package_metadata():
    """Verify package version and API exports."""
    assert drishti.__version__ == "0.1.0"
    assert hasattr(drishti, "analyze")
    assert hasattr(drishti, "profile")
    assert hasattr(drishti, "diagnose")
    assert hasattr(drishti, "benchmark")
    assert hasattr(drishti, "doctor")
    assert hasattr(drishti, "generate_report")


def test_doctor_api():
    """Verify drishti.doctor programmatic check."""
    report = drishti.doctor()
    assert isinstance(report, DoctorReport)
    assert report.schema == "drishti.doctor/v1"
    assert len(report.checks) >= 5
    assert report.python.get("version") is not None
    assert report.native_binary.get("found") is True
    # Test JSON serialization
    json_str = report.to_json()
    data = json.loads(json_str)
    assert data["schema"] == "drishti.doctor/v1"
    assert data["ok"] is True


def test_analyze_api(sample_kernel_path):
    """Verify drishti.analyze on sample Triton kernel."""
    report = drishti.analyze(sample_kernel_path, num_elements=65536, block_size=256)
    assert isinstance(report, AnalysisReport)
    assert report.schema == "drishti.analyze/v1"
    assert report.ok is True
    assert report.kernel_name == "fused_add_relu_kernel"
    # Verify stages
    assert report.stages.python_source_size > 0
    assert report.stages.ttir_size > 0
    assert report.stages.ttgir_size > 0
    assert report.stages.llvm_ir_size > 0
    assert report.stages.ptx_size > 0
    # Verify codegen metrics
    assert report.codegen.target_arch == "sm_86"
    assert report.codegen.num_warps == 4
    assert len(report.recommendations) > 0


def test_profile_api(sample_kernel_path):
    """Verify drishti.profile on sample kernel."""
    report = drishti.profile(sample_kernel_path, num_elements=65536, repeats=3)
    assert isinstance(report, ProfileReport)
    assert report.schema == "drishti.gpu_profile/v1"
    assert report.ok is True
    assert report.kernel_ms_avg >= 0.0
    assert report.gbps_effective >= 0.0


def test_diagnose_api(sample_kernel_path):
    """Verify drishti.diagnose findings and recommendations."""
    report = drishti.diagnose(sample_kernel_path, num_elements=65536)
    assert isinstance(report, DiagnosisReport)
    assert report.schema == "drishti.diagnosis/v1"
    assert report.ok is True
    assert len(report.findings) > 0
    assert report.primary_bottleneck != ""


def test_benchmark_api():
    """Verify drishti.benchmark 5-workload suite execution."""
    report = drishti.benchmark(num_elements=65536, repeats=3)
    assert isinstance(report, BenchmarkReport)
    assert report.schema == "drishti.benchmark/v1"
    assert report.ok is True
    assert len(report.workloads) == 5
    assert report.suite_summary.passed_workloads == 5
    assert report.suite_summary.rank_matched_workloads == 5


def test_html_report_generation(sample_kernel_path, tmp_path):
    """Verify standalone HTML report generation."""
    analysis = drishti.analyze(sample_kernel_path, num_elements=65536)
    out_html = str(tmp_path / "test_report.html")
    res = drishti.generate_report(analysis, output_path=out_html)
    assert isinstance(res, ReportOutput)
    assert os.path.isfile(res.html_path)
    assert res.file_size_bytes > 10000

    # Validate contents of generated HTML
    with open(res.html_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Dṛṣṭi Performance Intelligence Report" in content
    assert "Compiler Lowering Pipeline Flow" in content
    assert "TTIR" in content
    assert "TTGIR" in content
    assert "NVPTX" in content


def test_cli_execution():
    """Verify drishti console command execution via python -m drishti."""
    # Test --help
    p = subprocess.run([sys.executable, "-m", "drishti", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "doctor" in p.stdout
    assert "analyze" in p.stdout
    assert "profile" in p.stdout
    assert "diagnose" in p.stdout
    assert "benchmark" in p.stdout
    assert "report" in p.stdout

    # Test doctor
    p = subprocess.run([sys.executable, "-m", "drishti", "doctor", "--json"], capture_output=True, text=True)
    assert p.returncode == 0
    data = json.loads(p.stdout)
    assert data["schema"] == "drishti.doctor/v1"
