#!/bin/bash
# Run evaluation for Standard Model
python3 evaluate_blimp.py \
    --model_type decoder \
    --model_path /Users/begum/Downloads/standard \
    --data_path "/Users/begum/Downloads/blimp-master/blimp data"

# Generate plots
python3 plot_blimp_results.py
