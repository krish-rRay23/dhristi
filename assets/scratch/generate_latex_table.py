#!/usr/bin/env python3
"""
Generates journal_results_table.tex from journal_experiment_results.json
"""

import json

def generate_latex():
    with open("journal_experiment_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    latex = [
        "% Drishti Journal Benchmark Suite - Cross-System & Ablation Performance",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Cross-System Performance, Bandwidth, and Ablation Analysis across NVIDIA RTX 3050 Laptop GPU and NVIDIA Tesla T4.}",
        "\\label{tab:drishti_journal_results_complete}",
        "\\small",
        "\\begin{tabular}{lllrrrc}",
        "\\hline",
        "\\textbf{GPU Target} & \\textbf{Workload Domain} & \\textbf{Evaluation Target} & \\textbf{Latency (ms)} & \\textbf{Speedup} & \\textbf{BW (GB/s)} & \\textbf{Status} \\\\",
        "\\hline"
    ]

    current_gpu = None
    for r in results:
        gpu_short = "Tesla T4" if "Tesla T4" in r["hardware_device"] else "RTX 3050"
        if gpu_short != current_gpu:
            if current_gpu is not None:
                latex.append("\\hline")
            current_gpu = gpu_short

        target = r["ablation_id"] if r["ablation_id"] != "none" else r["system_id"]
        status = "PASS" if r["correctness"] else "FAIL"
        
        # Format speedup with bolding if best in workload
        sp_str = f"{r['speedup_vs_baseline']:.2f}\\times"
        if r['speedup_vs_baseline'] > 1.3:
            sp_str = f"\\textbf{{{sp_str}}}"

        latex.append(f"{gpu_short} & {r['workload_id']} & {target} & {r['latency_ms']:.4f} & {sp_str} & {r['vram_bandwidth_gbps']:.1f} & {status} \\\\")

    latex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    table_content = "\n".join(latex)
    with open("journal_results_table.tex", "w", encoding="utf-8") as f:
        f.write(table_content)
    print("Successfully generated journal_results_table.tex!")

if __name__ == "__main__":
    generate_latex()
