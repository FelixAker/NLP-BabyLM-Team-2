# BabyLM-Tiny: Evaluation & Metric Stability

This subdirectory contains the codebase for analyzing **metric stability** and **tokenizer-granularity bias** in low-resource language models. While the main repository focuses on model training and performance, this module provides the stability tools to ensure comparisons are scientifically fair.

## 🔍 The Problem: Tokenizer Bias
Standard BLiMP and GLUE evaluations often use "Per-Token Normalized" accuracy (Mean). However, our research shows that this metric is biased toward the specific tokenizer used during training (the "Anchor Point"). If you compare two models using different tokenizers, the "Mean" accuracy fluctuates by up to **7.7%**, making direct performance comparisons unreliable.

## 🛡️ The Solution: BPC and BPB
We demonstrate that **Bits-per-Character (BPC)** and **Bits-per-Byte (BPB)** are the only metrics in our suite that remain perfectly invariant to tokenizer granularity. This module provides the infrastructure to:
1.  Perform **Metric Stability Analysis** (evaluating a fixed model against 1k--40k simulated vocab sizes).
2.  Calculate **Balanced Metrics** (Sum, Mean, BPC, BPB) to identify length and granularity biases.
3.  Visualize the **Normalization Gap** between models with differing subword architectures.

## 📊 Key Experimental Outputs
*   **Morphology vs. Standard**: Using the stable BPB metric, we confirm a **60.19% vs. 54.66%** lead for morphology-aware tokenization.
*   **Categorical Stability Plots**: Visual proof of metric behavior across 8 vocabulary sizes (located in `plots/`).

## 📁 Module Structure
*   `plots/`: Visualizations of the stability analyses and category-level bias.
*   `results/`: Aligned data for 8 simulated vocabulary sizes (1k to 40k).
*   `metric_stability_analysis.py`: The core stability engine. It uses the exact same evaluation logic as the official `evaluate_blimp.py` but sweeps across multiple segmentations.
*   `scripts/evaluation/`: Logic for token imbalance and bias analysis.
*   `shell/`: Runner scripts for executing the stability suite.

## 🛠 Usage (Stability Mode)

To run the full 8-point aligned sweep for a model:
```bash
python3 metric_stability_analysis.py --model_path path/to/model --mode [morph|standard] --full
```

This generates:
1.  A JSON results file in `results/`.
2.  A categorical stability plot in `plots/` showing how BPB, BPC, Mean, and Sum interact.
