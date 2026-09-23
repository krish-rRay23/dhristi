#!/usr/bin/env python3
"""
Updates journal_results_table.tex & README.md based on audited dataset.
Preserves scientific integrity: verified GPU measurements are displayed with exact metrics;
unexecuted trials are explicitly marked as N/A with clear provenance explanations.
"""

import json

def update_docs():
    with open("journal_experiment_results.json", "r", encoding="utf-8") as f:
        ds = json.load(f)

    results = ds["results"]

    # 1. Generate journal_results_table.tex
    tex_lines = [
        "% Drishti Journal Research Benchmark Matrix - Audited Dataset",
        "% Provenance: Notebook Cell 3 (Tesla T4) & Local Audit Logs",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Cross-System Performance, Bandwidth, and Ablation Analysis across NVIDIA Tesla T4 and NVIDIA RTX 3050 Laptop GPU. Values marked N/A represent unexecuted trials to ensure strict empirical integrity.}",
        "\\label{tab:drishti_journal_results_audited}",
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
                tex_lines.append("\\hline")
            current_gpu = gpu_short

        target = r["ablation_id"] if r["ablation_id"] != "none" else r["system_id"]
        
        if r["provenance"]["status"] == "VERIFIED_GPU_MEASUREMENT":
            lat_str = f"{r['latency_ms']:.4f}"
            sp_val = r['speedup_vs_baseline']
            sp_str = f"{sp_val:.2f}\\times"
            if sp_val > 1.2:
                sp_str = f"\\textbf{{{sp_str}}}"
            bw_str = f"{r['vram_bandwidth_gbps']:.1f}"
            status = "PASS" if r["correctness"] else "FAIL"
        else:
            lat_str = "N/A"
            sp_str = "N/A"
            bw_str = "N/A"
            status = "UNEXECUTED"

        tex_lines.append(f"{gpu_short} & {r['workload_id']} & {target} & {lat_str} & {sp_str} & {bw_str} & {status} \\\\")

    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    with open("journal_results_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print("Updated journal_results_table.tex!")

if __name__ == "__main__":
    update_docs()
