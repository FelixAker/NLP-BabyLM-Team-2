#!/bin/bash
# Run token imbalance analysis for morphology_clean_fine_tuned model

echo "Running token imbalance analysis..."
python3 analyze_token_imbalance.py \
    --model_type decoder \
    --model_path /Users/begum/Downloads/morphology_clean_fine_tuned \
    --data_path "/Users/begum/Downloads/blimp-master/blimp data" \
    --output_file token_imbalance_results.json

echo ""
echo "Generating visualizations..."
python3 plot_token_imbalance.py \
    --input_file token_imbalance_results.json \
    --output_prefix token_imbalance

echo ""
echo "✓ Analysis complete!"
echo "Results saved to:"
echo "  - token_imbalance_results.json"
echo "  - token_imbalance_error_vs_delta.png"
echo "  - token_imbalance_category_table.png"
echo "  - token_imbalance_bias_reduction.png"
