#!/bin/bash
# Run token imbalance analysis for standard tokenizer model

echo "Running token imbalance analysis for standard model..."
python3 analyze_token_imbalance.py \
    --model_type decoder \
    --model_path /Users/begum/Downloads/standard \
    --data_path "/Users/begum/Downloads/blimp-master/blimp data" \
    --output_file token_imbalance_results_standard.json

echo ""
echo "Generating visualizations..."
python3 plot_token_imbalance.py \
    --input_file token_imbalance_results_standard.json \
    --output_prefix token_imbalance_standard

echo ""
echo "✓ Analysis complete!"
echo "Results saved to:"
echo "  - token_imbalance_results_standard.json"
echo "  - token_imbalance_standard_error_vs_delta.png"
echo "  - token_imbalance_standard_category_table.png"
echo "  - token_imbalance_standard_bias_reduction.png"
