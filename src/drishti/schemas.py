"""
Dṛṣṭi Data Models & Stable Schemas (v1)
========================================
Defines strongly-typed, serialization-stable data structures for compiler
pipeline artifacts, GPU profiling, diagnosis findings, benchmarks, and doctor health.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class BaseReport:
    """Base class providing dictionary & JSON serialization and attribute access."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report object to Python dictionary."""
        return asdict(self)  # type: ignore[call-overload]

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to a stable JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str, indent: int = 2) -> None:
        """Save report to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# -----------------------------------------------------------------------------
# 1. Doctor Schema (drishti.doctor/v1)
# -----------------------------------------------------------------------------

@dataclass
class CheckItem:
    name: str
    status: str  # "pass", "warn", "fail"
    message: str
    details: Optional[str] = None


@dataclass
class DoctorReport(BaseReport):
    schema: str = "drishti.doctor/v1"
    ok: bool = True
    timestamp: str = ""
    python: Dict[str, Any] = field(default_factory=dict)
    triton: Dict[str, Any] = field(default_factory=dict)
    llvm_mlir: Dict[str, Any] = field(default_factory=dict)
    cuda_driver: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    native_binary: Dict[str, Any] = field(default_factory=dict)
    nsight_compute: Dict[str, Any] = field(default_factory=dict)
    checks: List[CheckItem] = field(default_factory=list)
    summary: str = ""


# -----------------------------------------------------------------------------
# 2. Pipeline Stages & Analysis Schema (drishti.analyze/v1)
# -----------------------------------------------------------------------------

@dataclass
class PipelineStages:
    python_source: str = ""
    ttir: str = ""
    ttgir: str = ""
    llvm_ir: str = ""
    ptx: str = ""
    python_source_size: int = 0
    ttir_size: int = 0
    ttgir_size: int = 0
    llvm_ir_size: int = 0
    ptx_size: int = 0


@dataclass
class CodegenMetrics:
    num_warps: int = 4
    num_stages: int = 2
    shared_mem_bytes: int = 0
    register_count: int = 0
    target_arch: str = "sm_86"
    vector_width: str = "1 element/thread"
    v4_count: int = 0
    v2_count: int = 0
    b32_count: int = 0
    classification: str = "SCALAR_DECOMPOSED"


@dataclass
class GpuExecutionSummary:
    executed: bool = False
    correct: bool = False
    measured_kernel_ms: float = 0.0
    gbps_effective: float = 0.0
    tflops: float = 0.0
    device_name: str = ""


@dataclass
class CostModelSummary:
    predicted_kernel_ms: float = 0.0
    error_percent: float = 0.0
    is_accurate: bool = False
    bottleneck_regime: str = "memory-bound"
    theoretical_occupancy_pct: float = 100.0


@dataclass
class AnalysisReport(BaseReport):
    schema: str = "drishti.analyze/v1"
    ok: bool = True
    error: str = ""
    workload: str = ""
    kernel_name: str = ""
    source_file: Optional[str] = None
    num_elements: int = 65536
    block_size: int = 256
    grid_size: int = 256
    stages: PipelineStages = field(default_factory=PipelineStages)
    codegen: CodegenMetrics = field(default_factory=CodegenMetrics)
    gpu: GpuExecutionSummary = field(default_factory=GpuExecutionSummary)
    cost_model: CostModelSummary = field(default_factory=CostModelSummary)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# 3. GPU Profile Schema (drishti.gpu_profile/v1)
# -----------------------------------------------------------------------------

@dataclass
class ProfileReport(BaseReport):
    schema: str = "drishti.gpu_profile/v1"
    ok: bool = True
    error: str = ""
    backend: str = "cuda"
    device: Dict[str, Any] = field(default_factory=dict)
    kernel: str = ""
    num_elements: int = 1048576
    block_size: int = 256
    grid_size: int = 4096
    repeats: int = 10
    kernel_ms_avg: float = 0.0
    kernel_ms_min: float = 0.0
    h2d_ms: float = 0.0
    d2h_ms: float = 0.0
    wall_ms: float = 0.0
    gbps_effective: float = 0.0
    bytes_moved: int = 0
    tflops: float = 0.0
    correct: bool = True
    timing_source: str = "cuda-events"
    cupti_available: bool = False
    cupti_version: str = ""
    notes: str = ""
    ncu_metrics: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# 4. Diagnosis Schema (drishti.diagnosis/v1)
# -----------------------------------------------------------------------------

@dataclass
class DiagnosisFinding:
    rule_id: str
    title: str
    severity: str  # "critical", "warning", "note"
    confidence: float
    related_op: str = ""
    related_pass: str = ""
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    recommendation: str = ""


@dataclass
class DiagnosisReport(BaseReport):
    schema: str = "drishti.diagnosis/v1"
    ok: bool = True
    error: str = ""
    kernel: str = ""
    relation: str = "co-executed"
    findings: List[DiagnosisFinding] = field(default_factory=list)
    primary_bottleneck: str = ""
    recommendations: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# 5. Benchmark Schema (drishti.benchmark/v1)
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkWorkloadResult:
    name: str
    regime: str
    baseline_id: str
    predicted_winner: str
    actual_winner: str
    rank_matched: bool
    correct: bool
    baseline_ms: float
    best_ms: float
    speedup_factor: float
    prediction_error_pct: float


@dataclass
class BenchmarkSuiteSummary:
    total_workloads: int = 5
    passed_workloads: int = 5
    rank_matched_workloads: int = 5
    avg_prediction_error_pct: float = 0.0


@dataclass
class BenchmarkReport(BaseReport):
    schema: str = "drishti.benchmark/v1"
    ok: bool = True
    device: Dict[str, Any] = field(default_factory=dict)
    suite_summary: BenchmarkSuiteSummary = field(default_factory=BenchmarkSuiteSummary)
    workloads: List[BenchmarkWorkloadResult] = field(default_factory=list)


# -----------------------------------------------------------------------------
# 6. Report Output Schema (drishti.report/v1)
# -----------------------------------------------------------------------------

@dataclass
class ReportOutput(BaseReport):
    schema: str = "drishti.report/v1"
    ok: bool = True
    html_path: str = ""
    file_size_bytes: int = 0
    title: str = ""
    workload: str = ""
    target_device: str = ""
    generated_at: str = ""
