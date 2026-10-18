"""
Dṛṣṭi: ML Compiler and GPU Performance Analysis Framework
==========================================================
Python API for multi-level compiler analysis, GPU profiling,
bottleneck diagnosis, and reproducible benchmarking.

Example:
    >>> import drishti
    >>> report = drishti.analyze("examples/fused_add_relu.py")
    >>> print(report.codegen.classification)
    >>> drishti.generate_report(report, "fused_add_relu.html")
"""

from drishti.__version__ import __author__, __description__, __title__, __version__
from drishti.analyze import analyze
from drishti.benchmark import benchmark
from drishti.diagnose import diagnose
from drishti.doctor import doctor
from drishti.native import find_drishti_binary, set_binary_path
from drishti.profile import profile
from drishti.report import generate_report
from drishti.schemas import (
    AnalysisReport,
    BenchmarkReport,
    DiagnosisFinding,
    DiagnosisReport,
    DoctorReport,
    ProfileReport,
    ReportOutput,
)

__all__ = [
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "analyze",
    "profile",
    "diagnose",
    "benchmark",
    "doctor",
    "generate_report",
    "find_drishti_binary",
    "set_binary_path",
    "AnalysisReport",
    "ProfileReport",
    "DiagnosisReport",
    "BenchmarkReport",
    "DoctorReport",
    "ReportOutput",
    "DiagnosisFinding",
]
