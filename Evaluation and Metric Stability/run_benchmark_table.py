
import argparse
import subprocess
import pandas as pd
import json
import os
import sys

# Define your experiments here
# Format: (Model Name or Path, Tokenizer Path, Label)
EXPERIMENTS = [
    # Example:
    # ("sshleifer/tiny-gpt2", "sshleifer/tiny-gpt2", "TinyGPT2-Baseline"),
    # ("/path/to/my/model", "/Users/begum/Downloads/morphology_bpe", "Morphology-Model"),
]

def parse_output(output_str):
    """Parses the stdout from evaluate_blimp.py to extract metrics."""
    metrics = {}
    lines = output_str.split('\n')
    capture = False
    for line in lines:
        if "Average Accuracies:" in line:
            capture = True
            continue
        if capture and line.strip():
            try:
                key, val = line.split(':')
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return metrics

def run_evaluation(model_path, tokenizer_path, label):
    print(f"Running evaluation for {label}...")
    cmd = [
        sys.executable, "evaluate_blimp.py",
        "--model_type", "decoder",
        "--model_path", model_path,
        "--data_path", "/Users/begum/Downloads/blimp-master/blimp data", # Assuming this path based on context
    ]
    if tokenizer_path:
         cmd.extend(["--tokenizer_path", tokenizer_path])
         
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {label}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run BLiMP Benchmark Table")
    parser.add_argument("--add_model", action="append", help="Add a model to run. Format: model_path:tokenizer_path:label", default=[])
    args = parser.parse_args()

    experiments = EXPERIMENTS.copy()
    
    # Parse CLI arguments
    for item in args.add_model:
        parts = item.split(':')
        if len(parts) == 3:
            experiments.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 2:
            experiments.append((parts[0], parts[1], parts[0]))
        else:
            print(f"Invalid format for --add_model: {item}. Use model:tokenizer:label")

    if not experiments:
        print("No experiments defined. Use --add_model or edit EXPERIMENTS list in the script.")
        # Add a default example for demonstration if nothing provided
        print("Running default example with tiny-gpt2...")
        experiments.append(("sshleifer/tiny-gpt2", "sshleifer/tiny-gpt2", "TinyGPT2-Default"))

    results = []
    
    for model, tokenizer, label in experiments:
        metrics = {}
        
        # 1. Try BLiMP evaluation (only if model_path is a valid directory with a model)
        has_model = False
        if os.path.isdir(model) and (os.path.exists(os.path.join(model, "config.json")) or os.path.exists(os.path.join(model, "model.safetensors"))):
            has_model = True
        elif not model.startswith("/") and not model.startswith("."): # Likely a HF model
            has_model = True
            
        if has_model:
            blimp_metrics = run_evaluation(model, tokenizer, label)
            if blimp_metrics:
                metrics.update({
                    "sum": blimp_metrics.get("raw_log_prob", 0),
                    "mean": blimp_metrics.get("normalized_log_prob", 0),
                    "bpc": blimp_metrics.get("bpc", 0),
                    "bpb": blimp_metrics.get("bpb", 0)
                })
        
        # 2. Try to load tokenizer metrics from JSON
        # We expect files like results_morph_5000_tokenizer.json or results_baseline_tokenizer.json
        tokenizer_json = None
        if "Baseline" in label:
            tokenizer_json = "results_baseline_tokenizer.json"
        elif "Morph-" in label:
            size = label.split("-")[-1]
            tokenizer_json = f"results_morph_{size}_tokenizer.json"
            
        if tokenizer_json and os.path.exists(tokenizer_json):
            with open(tokenizer_json, "r") as f:
                t_metrics = json.load(f)
                metrics.update({
                    "Tokens/1k Chars": round(t_metrics.get("tokens_per_1000_chars", 0), 2),
                    "Avg Tokens/Word": round(t_metrics.get("avg_tokens_per_word", 0), 2),
                    "Morph Align %": f"{t_metrics.get('morph_alignment_score', 0)*100:.1f}%"
                })

        if metrics:
            metrics["Model"] = label
            results.append(metrics)

    if results:
        df = pd.DataFrame(results)
        # Fill missing values with "-"
        df = df.fillna("-")
        
        # Reorder columns
        cols = ["Model", "Tokens/1k Chars", "Avg Tokens/Word", "Morph Align %", "sum", "mean", "bpc", "bpb"]
        # Filter cols that exist
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        print("\n\n### Benchmark Results")
        try:
            markdown_table = df.to_markdown(index=False)
            print(markdown_table)
            with open("benchmark_results.md", "w") as f:
                f.write(markdown_table)
        except ImportError:
            print(df.to_string(index=False))
            # Manual simple markdown table generation
            header = "| " + " | ".join(df.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = []
            for _, row in df.iterrows():
                rows.append("| " + " | ".join([str(val) for val in row]) + " |")
            markdown_table = "\n".join([header, separator] + rows)
            with open("benchmark_results.md", "w") as f:
                f.write(markdown_table)
        
        print("\nResults saved to benchmark_results.md")

if __name__ == "__main__":
    main()
