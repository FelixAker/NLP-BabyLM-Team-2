#!/bin/bash

# Configuration
SIZES=(5000 10000 30000)
DATA_PATH="/Users/begum/Downloads/blimp-master/blimp data"
BASELINE_MODEL="/Users/begum/Downloads/morphology_clean_fine_tuned"
BASELINE_TOKENIZER="/Users/begum/Downloads/morphology_clean_fine_tuned"

echo "Starting Phase 7B: Tokenizer Granularity Sweep..."

# 0. Baseline Analysis
echo "=================================================="
echo "PROCESSING BASELINE: morphology_clean_fine_tuned"
echo "=================================================="
if [ ! -f "results_baseline_bias.json" ]; then
    python3 analyze_token_imbalance.py --model_type decoder --model_path "${BASELINE_MODEL}" --data_path "${DATA_PATH}" --output_file results_baseline_bias.json
fi
if [ ! -f "results_baseline_blimp.json" ]; then
    python3 evaluate_blimp.py --model_type decoder --model_path "${BASELINE_MODEL}" --data_path "${DATA_PATH}" --tokenizer_path "${BASELINE_TOKENIZER}"
    mv blimp_results.json results_baseline_blimp.json
fi
python3 evaluate_tokenizers.py --tokenizer "${BASELINE_TOKENIZER}" --output results_baseline_tokenizer.json

for size in "${SIZES[@]}"; do
    echo "=================================================="
    echo "PROCESSING VOCAB SIZE: ${size}"
    echo "=================================================="
    
    TOKENIZER_PATH="data/tokenizers_morph/${size}/tokenizer.model"
    OUTPUT_DIR="models_morph/${size}"
    
    # Set paths based on whether we use the baseline or a new variant
    if [ "${size}" -eq 10000 ]; then
        EVAL_MODEL_PATH="${BASELINE_MODEL}"
        EVAL_TOKENIZER_PATH="${BASELINE_TOKENIZER}"
        echo "[1/3] Skipping training for size 10000 (using baseline)."
    else
        EVAL_MODEL_PATH="${OUTPUT_DIR}"
        EVAL_TOKENIZER_PATH="${TOKENIZER_PATH}"
        # 1. Training (Option A: Retrain per size)
        if [ ! -d "${OUTPUT_DIR}" ] || [ ! -f "${OUTPUT_DIR}/config.json" ]; then
            echo "[1/3] Training model for size ${size}..."
            python3 train_model_variants.py --vocab_size ${size} --tokenizer_path ${TOKENIZER_PATH} --output_dir ${OUTPUT_DIR} --epochs 10
        else
            echo "[1/3] Model already exists for size ${size}."
        fi
    fi
    
    # 2. Evaluation & Analysis
    echo "[2/3] Analyzing Token Imbalance for size ${size}..."
    python3 analyze_token_imbalance.py --model_type decoder --model_path ${EVAL_MODEL_PATH} --data_path "${DATA_PATH}" --output_file results_morph_${size}_bias.json
    
    echo "[3/3] Evaluating BLiMP for size ${size}..."
    if [ ! -f "results_morph_${size}_blimp.json" ]; then
        python3 evaluate_blimp.py --model_type decoder --model_path ${EVAL_MODEL_PATH} --data_path "${DATA_PATH}" --tokenizer_path ${EVAL_TOKENIZER_PATH}
        mv blimp_results.json results_morph_${size}_blimp.json
    fi

    echo "[4/4] Evaluating Tokenizer metrics for size ${size}..."
    python3 evaluate_tokenizers.py --tokenizer "data/tokenizers_morph/${size}" --output results_morph_${size}_tokenizer.json
    
done

echo "=================================================="
echo "GENERATING COMPARISON TABLE"
echo "=================================================="

# Build the run_benchmark_table command
CMD="python3 run_benchmark_table.py"
# Add Baseline first
CMD="${CMD} --add_model ${BASELINE_MODEL}:${BASELINE_TOKENIZER}:Baseline-Clean"
for size in "${SIZES[@]}"; do
    CMD="${CMD} --add_model models_morph/${size}:data/tokenizers_morph/${size}/tokenizer.model:Morph-${size}"
done

# Run the final table generation
eval $CMD

echo "Experiment suite complete. Results are in benchmark_results.md"
