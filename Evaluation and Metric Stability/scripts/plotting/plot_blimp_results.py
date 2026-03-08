import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_results(json_path="blimp_results.json"):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Please run the evaluation first.")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Average Accuracy by Metric ---
    averages = data.get("averages", {})
    if not averages and "subsets" in data:
         # recalcluate averages if missing but subsets exist
         metric_names = ["raw_log_prob", "normalized_log_prob", "bpc", "bpb"]
         aggregated = {m: [] for m in metric_names}
         for res in data["subsets"].values():
             for m in metric_names:
                 aggregated[m].append(res[m])
         averages = {m: sum(vals)/len(vals) for m, vals in aggregated.items()}

    if averages:
        metrics = list(averages.keys())
        # Clean up names for display
        display_names = {
            "raw_log_prob": "Raw Log-Prob (Sum)",
            "normalized_log_prob": "Per-token Normalized Log-Prob (Mean)",
            "bpc": "Bits-per-character (BPC)",
            "bpb": "Bits-per-byte (BPB)"
        }
        labels = [display_names.get(m, m) for m in metrics]
        values = list(averages.values())

        plt.figure(figsize=(10, 6))
        # Use a nice color palette
        colors = sns.color_palette("husl", len(metrics))
        bars = plt.bar(labels, values, color=colors, alpha=0.8)
        
        plt.title('Average BLiMP Accuracy by Metric (Standard Baseline Model)', fontsize=16)
        plt.ylabel('Average Accuracy', fontsize=12)
        plt.xlabel('Evaluation Metric', fontsize=12)
        plt.ylim(0.4, 0.6) # Focus on the relevant range (random is 0.5)
        plt.xticks(rotation=45, ha='right')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('blimp_average_accuracy.png', dpi=300)
        print("Saved blimp_average_accuracy.png")
        plt.close()

    # --- Plot 2: Difference Plot (BPB vs Normalized Log Prob) ---
    subsets = data.get("subsets", {})
    if subsets:
        diff_data = []
        for name, res in subsets.items():
            # Difference: BPB Accuracy - Normalized Log Prob Accuracy
            # Or usually: "Metric A" - "Metric B"
            # The prompt mentions "Bit-per-Byte vs Per-token Normalized Log-Prob"
            # Let's assume (BPB - Norm)
            diff = res["bpb"] - res["normalized_log_prob"]
            diff_data.append({"subset": name, "diff": diff})
        
        # Sort by difference
        diff_data.sort(key=lambda x: x["diff"], reverse=True)
        
        df = pd.DataFrame(diff_data)
        
        plt.figure(figsize=(15, 8))
        
        # Color positive differences blue, negative red
        colors = ['skyblue' if x >= 0 else 'lightcoral' for x in df['diff']]
        
        sns.barplot(x="subset", y="diff", data=df, palette=colors)
        
        plt.axhline(0, color='grey', linewidth=0.8)
        plt.title('Difference in BLIMP Accuracy: Bits-per-Byte vs. Per-token Normalized Log-Prob', fontsize=16)
        plt.xlabel('BLiMP Subset', fontsize=12)
        plt.ylabel('Delta Accuracy (BPB - Mean)', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('blimp_accuracy_difference.png', dpi=300)
        print("Saved blimp_accuracy_difference.png")
        plt.close()

    # --- Plot 3: Sensitivity (Absolute Difference: Mean vs BPB) ---
    if subsets:
        sensitivity_data = []
        for name, res in subsets.items():
            # Absolute Difference: |Normalized Log Prob - BPB|
            diff = abs(res["normalized_log_prob"] - res["bpb"])
            sensitivity_data.append({"subset": name, "diff": diff})
        
        # Sort by absolute difference descending
        sensitivity_data.sort(key=lambda x: x["diff"], reverse=True)
        
        df_sens = pd.DataFrame(sensitivity_data)
        
        plt.figure(figsize=(15, 8))
        
        # Purple color match
        sns.barplot(x="subset", y="diff", data=df_sens, color='purple')
        
        plt.title('BLIMP Subset Sensitivity to Metric Choice (Mean vs. BPB)', fontsize=16)
        plt.ylabel('Absolute Difference in Accuracy (|Mean - BPB|)', fontsize=12)
        plt.xlabel('BLIMP Subset', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('blimp_sensitivity.png', dpi=300)
        print("Saved blimp_sensitivity.png")
        plt.close()

if __name__ == "__main__":
    plot_results()
