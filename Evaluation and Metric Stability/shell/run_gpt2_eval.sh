#!/bin/bash
# Run evaluation for sshleifer/tiny-gpt2 Model
python3 evaluate_blimp.py \
    --model_type decoder \
    --model_path sshleifer/tiny-gpt2 \
    --data_path "/Users/begum/Downloads/blimp-master/blimp data"

# Generate plots (optional)
python3 plot_blimp_results.py