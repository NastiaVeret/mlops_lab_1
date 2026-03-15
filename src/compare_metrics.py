import json
import os

def create_markdown_comparison(current_path="metrics.json", baseline_path="baseline/metrics.json", output_path="metrics_diff.md"):
    if not os.path.exists(current_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Поточні метрики не знайдені (`metrics.json`).\n")
        return

    with open(current_path, "r", encoding="utf-8") as f:
        current_metrics = json.load(f)

    if not os.path.exists(baseline_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Файл `baseline/metrics.json` не знайдено. Ось поточні метрики:\n\n")
            f.write("| Метрика | Значення |\n")
            f.write("|---------|----------|\n")
            for k, v in current_metrics.items():
                f.write(f"| {k} | {v:.4f} |\n")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("| Метрика | Baseline (main) | Поточний PR | Різниця (Δ) |\n")
        f.write("|---------|-----------------|-------------|-------------|\n")
        
        all_keys = set(current_metrics.keys()).union(set(baseline_metrics.keys()))
        for k in sorted(all_keys):
            curr_val = current_metrics.get(k, 0.0)
            base_val = baseline_metrics.get(k, 0.0)
            diff = curr_val - base_val
            
            # Formating difference
            if diff > 0:
                diff_str = f"+{diff:.4f} 🟢"
            elif diff < 0:
                diff_str = f"{diff:.4f} 🔴"
            else:
                diff_str = "0.0000 ⚪"
            
            f.write(f"| {k} | {base_val:.4f} | {curr_val:.4f} | {diff_str} |\n")

if __name__ == "__main__":
    create_markdown_comparison()
