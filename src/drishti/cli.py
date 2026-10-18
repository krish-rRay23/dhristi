"""
Dṛṣṭi Command-Line Interface (CLI) Entrypoint
==============================================
Provides unified console commands for Dṛṣṭi:
- drishti doctor
- drishti analyze <kernel.py>
- drishti profile <kernel.py>
- drishti diagnose <kernel.py>
- drishti benchmark
- drishti report <json_or_kernel>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from typing import List, Optional

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from drishti.__version__ import __version__
from drishti.analyze import analyze, format_analysis_report
from drishti.benchmark import benchmark, format_benchmark_report
from drishti.diagnose import diagnose, format_diagnosis_report
from drishti.doctor import doctor, format_doctor_report
from drishti.profile import format_profile_report, profile
from drishti.report import generate_report


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="drishti",
        description="Dṛṣṭi: ML Compiler and GPU Performance Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  drishti doctor
  drishti analyze examples/fused_add_relu.py --show-ir
  drishti profile examples/fused_add_relu.py --repeats 10
  drishti diagnose examples/fused_add_relu.py
  drishti benchmark
  drishti report analysis.json --output report.html

For more info, visit: https://github.com/drishti-ml/drishti
        """,
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"Dṛṣṭi CLI v{__version__}",
        help="Show version information and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        metavar="<command>",
    )

    # 1. doctor
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Inspect environment, Python, Triton, LLVM/MLIR, CUDA, GPU, and Nsight Compute.",
    )
    p_doctor.add_argument("--json", action="store_true", help="Output machine-readable JSON (drishti.doctor/v1).")

    # 2. analyze
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Run compiler lowering analysis across AST -> TTIR -> TTGIR -> LLVM -> PTX.",
    )
    p_analyze.add_argument("kernel", help="Path to .py kernel file or registered workload name.")
    p_analyze.add_argument("--block", type=int, default=256, help="GPU block size / threads per block (default: 256).")
    p_analyze.add_argument("--n", type=int, default=65536, help="Vector length / total elements (default: 65536).")
    p_analyze.add_argument("--repeats", type=int, default=5, help="Execution repeats for verification (default: 5).")
    p_analyze.add_argument("--show-ir", action="store_true", help="Print captured TTIR, TTGIR, LLVM IR, and PTX.")
    p_analyze.add_argument("--no-gpu", action="store_true", help="Skip live GPU execution and verification.")
    p_analyze.add_argument("--json", action="store_true", help="Output machine-readable JSON (drishti.analyze/v1).")
    p_analyze.add_argument("-o", "--output", type=str, help="Save report to file (.json or .html based on format).")
    p_analyze.add_argument("--html", type=str, help="Generate standalone HTML performance report directly.")

    # 3. profile
    p_profile = subparsers.add_parser(
        "profile",
        help="Profile kernel execution on GPU measuring latency, bandwidth, and transfer overheads.",
    )
    p_profile.add_argument("kernel", nargs="?", default=None, help="Path to .py kernel, workload name, or empty for vecadd.")
    p_profile.add_argument("--block", type=int, default=256, help="GPU block size (default: 256).")
    p_profile.add_argument("--n", type=int, default=1048576, help="Total elements (default: 1048576).")
    p_profile.add_argument("--repeats", type=int, default=10, help="Timed kernel iterations (default: 10).")
    p_profile.add_argument("--ncu", action="store_true", help="Attach NVIDIA Nsight Compute CLI profiler if installed.")
    p_profile.add_argument("--json", action="store_true", help="Output machine-readable JSON (drishti.gpu_profile/v1).")
    p_profile.add_argument("-o", "--output", type=str, help="Save JSON profile results to file.")

    # 4. diagnose
    p_diagnose = subparsers.add_parser(
        "diagnose",
        help="Diagnose hardware and compiler performance bottlenecks with actionable recommendations.",
    )
    p_diagnose.add_argument("kernel", nargs="?", default=None, help="Path to .py kernel or workload name.")
    p_diagnose.add_argument("--block", type=int, default=256, help="GPU block size (default: 256).")
    p_diagnose.add_argument("--n", type=int, default=65536, help="Total elements (default: 65536).")
    p_diagnose.add_argument("--repeats", type=int, default=5, help="Iterations (default: 5).")
    p_diagnose.add_argument("--json", action="store_true", help="Output machine-readable JSON (drishti.diagnosis/v1).")
    p_diagnose.add_argument("-o", "--output", type=str, help="Save JSON diagnosis results to file.")

    # 5. benchmark
    p_benchmark = subparsers.add_parser(
        "benchmark",
        help="Run Dṛṣṭi Generalization Benchmark Suite (5 Workloads) or Journal Experiments.",
    )
    p_benchmark.add_argument("--workload", type=str, default=None, help="Filter specific workload (gemm, reduction, fusion, etc.).")
    p_benchmark.add_argument("--n", type=int, default=65536, help="Elements per workload (default: 65536).")
    p_benchmark.add_argument("--repeats", type=int, default=5, help="Iterations (default: 5).")
    p_benchmark.add_argument("--journal", action="store_true", help="Run full 48-experiment journal research suite.")
    p_benchmark.add_argument("--json", action="store_true", help="Output machine-readable JSON (drishti.benchmark/v1).")
    p_benchmark.add_argument("-o", "--output", type=str, help="Save JSON benchmark results to file.")

    # 6. report
    p_report = subparsers.add_parser(
        "report",
        help="Generate a polished standalone HTML report showing Triton -> TTIR -> TTGIR -> LLVM -> PTX -> GPU.",
    )
    p_report.add_argument("input", help="Path to input JSON analysis report or .py kernel file.")
    p_report.add_argument("-o", "--output", type=str, default=None, help="Path to save HTML report (default: drishti_report_<name>.html).")
    p_report.add_argument("--open", action="store_true", help="Automatically open generated HTML report in default browser.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI handler."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        # Command 1: doctor
        if args.command == "doctor":
            doc = doctor()
            if args.json:
                print(doc.to_json())
            else:
                print(format_doctor_report(doc))
            return 0 if doc.ok else 1

        # Command 2: analyze
        elif args.command == "analyze":
            rep = analyze(
                kernel_source=args.kernel,
                block_size=args.block,
                num_elements=args.n,
                repeats=args.repeats,
                verify_gpu=not args.no_gpu,
            )
            if args.json:
                json_str = rep.to_json()
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(json_str)
                    print(f"Analysis saved to {args.output}")
                else:
                    print(json_str)
            else:
                print(format_analysis_report(rep, show_ir=args.show_ir))

            if args.html:
                out = generate_report(rep, output_path=args.html)
                print(f"[HTML Report Generated]: {out.html_path} ({out.file_size_bytes:,} bytes)")

            return 0 if rep.ok else 1

        # Command 3: profile
        elif args.command == "profile":
            prof = profile(
                kernel_source=args.kernel,
                num_elements=args.n,
                block_size=args.block,
                repeats=args.repeats,
                use_ncu=args.ncu,
            )
            if args.json:
                json_str = prof.to_json()
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(json_str)
                    print(f"Profile saved to {args.output}")
                else:
                    print(json_str)
            else:
                print(format_profile_report(prof))
            return 0 if prof.ok else 1

        # Command 4: diagnose
        elif args.command == "diagnose":
            diag = diagnose(
                kernel_source=args.kernel,
                num_elements=args.n,
                block_size=args.block,
                repeats=args.repeats,
            )
            if args.json:
                json_str = diag.to_json()
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(json_str)
                    print(f"Diagnosis saved to {args.output}")
                else:
                    print(json_str)
            else:
                print(format_diagnosis_report(diag))
            return 0 if diag.ok else 1

        # Command 5: benchmark
        elif args.command == "benchmark":
            bench = benchmark(
                workload=args.workload,
                num_elements=args.n,
                repeats=args.repeats,
                journal=args.journal,
            )
            if args.json:
                json_str = bench.to_json()
                if args.output:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(json_str)
                    print(f"Benchmark results saved to {args.output}")
                else:
                    print(json_str)
            else:
                print(format_benchmark_report(bench))
            return 0 if bench.ok else 1

        # Command 6: report
        elif args.command == "report":
            input_target = args.input
            # If the user passed a .py kernel directly to report, analyze it first
            if input_target.endswith(".py") and os.path.isfile(input_target):
                print(f"Compiling and analyzing kernel: {input_target} ...")
                analysis_data = analyze(input_target)
                out = generate_report(analysis_data, output_path=args.output)
            else:
                out = generate_report(input_target, output_path=args.output)

            print(f"================================================================================")
            print(f"               DṚṢṬI STANDALONE HTML PERFORMANCE REPORT GENERATED               ")
            print(f"================================================================================")
            print(f"Report File  : {out.html_path}")
            print(f"File Size    : {out.file_size_bytes:,} bytes")
            print(f"Workload     : {out.workload}")
            print(f"Target Device: {out.target_device}")
            print(f"Pipeline Flow: Triton AST -> TTIR -> TTGIR -> LLVM IR -> NVPTX -> GPU")
            print(f"================================================================================")

            if args.open:
                webbrowser.open(f"file://{out.html_path}")
            return 0

    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
