import json

def clean_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Cleaned outputs from {nb_path}")

if __name__ == "__main__":
    clean_notebook("notebooks/Drishti_Journal_Benchmark_Suite.ipynb")
