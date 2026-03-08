#!/bin/bash
# Run evaluation for Morphology Clean Fine-Tuned Model
python3 evaluate_blimp.py \
    --model_type decoder \
    --model_path /Users/begum/Downloads/morphology_clean_fine_tuned \
    --data_path "/Users/begum/Downloads/blimp-master/blimp data"

# Generate plots (optional)
python3 plot_blimp_results.py