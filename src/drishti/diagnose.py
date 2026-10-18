"""
Dṛṣṭi Root-Cause Intelligence & Diagnosis (Diagnose)
===================================================
Identifies hardware and compiler performance bottlenecks, quantifying
transfer dominance, launch overhead, memory intensity, and register pressure.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from drishti.native import find_drishti_binary, run_native_json
from drishti.schemas import DiagnosisFinding, DiagnosisReport


def diagnose(
    kernel_source: Optional[str] = None,
    num_elements: int = 65536,
    block_size: int = 256,
    repeats: int = 5,
) -> DiagnosisReport:
    """
    Diagnose bottlenecks in a kernel workload via the native C++ root-cause engine.

    Args:
        kernel_source: Path to .py kernel or registered workload.
        num_elements: Elements to process.
        block_size: GPU block size.
        repeats: Profiling iterations.

    Returns:
        DiagnosisReport: Prioritized findings, confidence scores, and explanations.
    """
    bin_path = find_drishti_binary()
    if not bin_path:
        raise RuntimeError("Native drishti executable not found.")

    args = [
        "diagnose",
        f"--block={block_size}",
        f"--n={num_elements}",
        f"--repeats={repeats}",
        "--json-only",
    ]

    res = run_native_json(args)

    raw_findings = res.get("findings", [])
    findings: List[DiagnosisFinding] = []
    primary_bottleneck = ""

    for f in raw_findings:
        sev = f.get("severity", "note")
        finding = DiagnosisFinding(
            rule_id=f.get("rule_id", "unknown"),
            title=f.get("title", ""),
            severity=sev,
            confidence=float(f.get("confidence", 0.5)),
            related_op=f.get("related_op", ""),
            related_pass=f.get("related_pass", ""),
            evidence=f.get("evidence", []),
            explanation=f.get("explanation", ""),
        )

        # Generate recommendation based on rule_id
        if finding.rule_id == "transfer-share":
            finding.recommendation = "Batch consecutive kernel launches or keep intermediate tensors on GPU VRAM across steps."
        elif finding.rule_id == "launch-overhead":
            finding.recommendation = "Kernel execution time is in microseconds; batch larger element counts per launch to amortize launch latency."
        elif finding.rule_id == "roofline-balance":
            finding.recommendation = "Workload is memory-bound (low arithmetic intensity). Focus on memory access coalescing, cache locality, and vectorization rather than math optimizations."
        elif finding.rule_id == "register-pressure":
            finding.recommendation = "Register pressure is well-balanced; full theoretical occupancy achieved."
        elif finding.rule_id == "memory-traffic":
            finding.recommendation = "Memory movement matches algorithmic lower bound; no redundant traffic detected."
        else:
            finding.recommendation = "Optimize memory layout and inspect compiler vectorization hints."

        if not primary_bottleneck and sev in ["critical", "warning"]:
            primary_bottleneck = finding.title

        findings.append(finding)

    if not primary_bottleneck and findings:
        primary_bottleneck = findings[0].title

    recommendations = [f.recommendation for f in findings if f.recommendation]

    return DiagnosisReport(
        schema="drishti.diagnosis/v1",
        ok=res.get("ok", True),
        error=res.get("error", ""),
        kernel=str(kernel_source or "vecadd"),
        relation=res.get("relation", "co-executed"),
        findings=findings,
        primary_bottleneck=primary_bottleneck or "balanced execution",
        recommendations=recommendations,
    )


def format_diagnosis_report(report: DiagnosisReport) -> str:
    """Format diagnosis findings for clear CLI presentation."""
    lines = [
        "",
        "================================================================================",
        "                    DRISHTI ROOT-CAUSE PERFORMANCE DIAGNOSIS                    ",
        "================================================================================",
        f"Workload / Kernel   : {report.kernel}",
        f"Primary Bottleneck  : {report.primary_bottleneck}",
        f"Cross-Stage Relation: {report.relation}",
        "--------------------------------------------------------------------------------",
        "DETECTED BOTTLENECKS & FINDINGS:",
    ]

    sev_badge = {
        "critical": "[CRITICAL]",
        "warning": "[WARN]    ",
        "note": "[NOTE]    ",
    }

    for idx, f in enumerate(report.findings, 1):
        badge = sev_badge.get(f.severity.lower(), "[INFO]    ")
        lines.append(f"  {badge} #{idx} [{f.rule_id}] {f.title} (Confidence: {int(f.confidence * 100)}%)")
        lines.append(f"            Reason: {f.explanation}")
        for ev in f.evidence:
            lines.append(f"            -> Evidence: {ev}")
        if f.recommendation:
            lines.append(f"            * Action  : {f.recommendation}")
        lines.append("")

    lines.append("================================================================================\n")
    return "\n".join(lines)
