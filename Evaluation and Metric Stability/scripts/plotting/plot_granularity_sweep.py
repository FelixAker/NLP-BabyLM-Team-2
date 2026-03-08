import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_data(sizes, baseline_label="Baseline-Clean"):
    all_data = []
    
    # Load baseline
    if os.path.exists("results_baseline_blimp.json") and os.path.exists("results_baseline_tokenizer.json"):
        with open("results_baseline_blimp.json", "r") as f:
            blimp = json.load(f)["averages"]
        with open("results_baseline_tokenizer.json", "r") as f:
            tokens = json.load(f)
        
        all_data.append({
            "label": baseline_label,
            "vocab_size": tokens["vocab_size"],
            "per_token": blimp["normalized_log_prob"],
            "bpb": blimp["bpb"],
            "avg_tokens_per_word": tokens["avg_tokens_per_word"],
            "tokens_per_1k_chars": tokens["tokens_per_1000_chars"],
            "delta": blimp["normalized_log_prob"] - blimp["bpb"]
        })

    # Load sweep
    for size in sizes:
        blimp_path = f"results_morph_{size}_blimp.json"
        token_path = f"results_morph_{size}_tokenizer.json"
        
        if os.path.exists(blimp_path) and os.path.exists(token_path):
            with open(blimp_path, "r") as f:
                blimp = json.load(f)["averages"]
            with open(token_path, "r") as f:
                tokens = json.load(f)
                
            all_data.append({
                "label": f"Morph-{size}",
                "vocab_size": size,
                "per_token": blimp["normalized_log_prob"],
                "bpb": blimp["bpb"],
                "avg_tokens_per_word": tokens["avg_tokens_per_word"],
                "tokens_per_1k_chars": tokens["tokens_per_1000_chars"],
                "delta": blimp["normalized_log_prob"] - blimp["bpb"]
            })
            
    return pd.DataFrame(all_data)

def plot_granularity(df):
    if df.empty:
        print("No data found for plotting.")
        return

    # Sort by vocab size
    df = df.sort_values("vocab_size")
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Accuracy vs Vocab Size (per metric)
    plt.figure(figsize=(10, 6))
    plt.plot(df["vocab_size"], df["per_token"], 'o-', label="Per-token Normalized", linewidth=2)
    plt.plot(df["vocab_size"], df["bpb"], 's-', label="Bits-per-Byte (BPB)", linewidth=2)
    plt.xscale('log')
    plt.title('BLiMP Accuracy vs. Vocab Size', fontsize=16)
    plt.xlabel('Vocabulary Size (Log Scale)', fontsize=12)
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('granularity_accuracy_vs_size.png', dpi=300)
    plt.close()

    # Plot 2: Δ(per-token − BPB) vs Vocab Size
    plt.figure(figsize=(10, 6))
    plt.plot(df["vocab_size"], df["delta"], 'D-', color='purple', linewidth=2)
    plt.xscale('log')
    plt.axhline(0, color='grey', linestyle='--', alpha=0.7)
    plt.title('Normalization Gap vs. Vocab Size', fontsize=16)
    plt.xlabel('Vocabulary Size (Log Scale)', fontsize=12)
    plt.ylabel('Δ (Per-token - BPB)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('granularity_delta_vs_size.png', dpi=300)
    plt.close()

    # Plot 3: Avg tokens per word vs Vocab Size
    plt.figure(figsize=(10, 6))
    plt.plot(df["vocab_size"], df["avg_tokens_per_word"], 'v-', color='green', linewidth=2)
    plt.xscale('log')
    plt.title('Sequence Length vs. Vocab Size', fontsize=16)
    plt.xlabel('Vocabulary Size (Log Scale)', fontsize=12)
    plt.ylabel('Avg Tokens per Word', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('granularity_length_vs_size.png', dpi=300)
    plt.close()
    
    print("Saved Phase 7B plots: granularity_accuracy_vs_size.png, granularity_delta_vs_size.png, granularity_length_vs_size.png")

if __name__ == "__main__":
    SIZES = [5000, 10000, 30000]
    df = load_data(SIZES)
    plot_granularity(df)
