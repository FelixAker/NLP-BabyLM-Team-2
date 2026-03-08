#!/bin/bash
# Install missing dependencies
pip install sentencepiece matplotlib seaborn pandas

# Run evaluation (saves results to blimp_results.json)
python3 evaluate_blimp.py \
    --model_type decoder \
    --model_path /Users/begum/Downloads/morphology_clean_tokenizer/morphology_clean_tokenizer \
    --data_path "/Users/begum/Downloads/blimp-master/blimp data"

# Generate plots
python3 plot_blimp_results.py
