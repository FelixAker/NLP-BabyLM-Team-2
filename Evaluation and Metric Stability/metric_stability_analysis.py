import argparse
import torch
import datasets
import os
import math
import json
from transformers import AutoModelForCausalLM, DebertaV2Tokenizer
from functools import partial
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def tokenize_decoder(examples, tokenizer):
    batch = {
        "good_inputs": [],
        "bad_inputs": [],
        "good_labels": [],
        "bad_labels": [],
        "good_chars": [],
        "bad_chars": [],
        "good_bytes": [],
        "bad_bytes": [],
    }

    for i in range(len(examples['sentence_good'])):
        good_tokens = tokenizer.encode(examples['sentence_good'][i])
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i])
        batch["good_inputs"].append(good_tokens[:-1])
        batch["bad_inputs"].append(bad_tokens[:-1])
        batch["good_labels"].append(good_tokens[1:])
        batch["bad_labels"].append(bad_tokens[1:])
        
        batch["good_chars"].append([len(examples['sentence_good'][i])])
        bad_chars = len(examples['sentence_bad'][i])
        batch["bad_chars"].append([bad_chars])
        batch["good_bytes"].append([len(examples['sentence_good'][i].encode('utf-8'))])
        batch["bad_bytes"].append([len(examples['sentence_bad'][i].encode('utf-8'))])

    return batch

def padding_collate_fn(batch, max_len=1024, left_padding=False):
    padded_batch = {}
    for key in batch[0]:
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            if left_padding:
                padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])
            else:
                padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch

def evaluate_decoder_sweep(model, dataloader, sweep_tokenizers):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # key: vocab_size, val: metrics dict
    sweep_metrics = {size: {
        "raw_log_prob": {"correct": 0.0, "total": 0.0},
        "normalized_log_prob": {"correct": 0.0, "total": 0.0},
        "bpc": {"correct": 0.0, "total": 0.0},
        "bpb": {"correct": 0.0, "total": 0.0}
    } for size in sweep_tokenizers.keys()}

    with torch.no_grad():
        for batch in dataloader:
            good_inputs = batch['good_inputs'].to(device=DEVICE)
            bad_inputs = batch['bad_inputs'].to(device=DEVICE)
            good_labels = batch['good_labels'].to(device=DEVICE)
            bad_labels = batch['bad_labels'].to(device=DEVICE)

            good_chars = batch['good_chars'].to(device=DEVICE).float().view(-1)
            bad_chars = batch['bad_chars'].to(device=DEVICE).float().view(-1)
            good_bytes = batch['good_bytes'].to(device=DEVICE).float().view(-1)
            bad_bytes = batch['bad_bytes'].to(device=DEVICE).float().view(-1)

            # Get logits once for the anchor model
            good_logits = model(good_inputs, attention_mask=good_inputs != 0).logits
            bad_logits = model(bad_inputs, attention_mask=bad_inputs != 0).logits
            
            good_token_loss = loss_fn(good_logits.view(-1, good_logits.shape[-1]), good_labels.view(-1))
            bad_token_loss = loss_fn(bad_logits.view(-1, bad_logits.shape[-1]), bad_labels.view(-1))

            good_token_loss = good_token_loss.view(good_inputs.shape[0], -1)
            bad_token_loss = bad_token_loss.view(bad_inputs.shape[0], -1)
            
            good_mask = (good_labels != -100).float()
            bad_mask = (bad_labels != -100).float()
            
            good_nll_sum = (good_token_loss * good_mask).sum(dim=1)
            bad_nll_sum = (bad_token_loss * bad_mask).sum(dim=1)
            
            good_log_prob = -good_nll_sum
            bad_log_prob = -bad_nll_sum

            # Now iterate over sweep tokenizers to get DIFFERENT normalization lengths
            # NOTE: We assume the batch was tokenized by the ANCHOR tokenizer for model inputs.
            # For BPC/BPB/Sum, the results are invariant to the specific tokenizer used.
            # Only the Per-token metric changes.
            
            # To get the Per-token lengths for OTHER tokenizers, we need to re-tokenize.
            # This is slightly slow but necessary for the stability.
            
            for size, tok in sweep_tokenizers.items():
                # We need the original text. We'll reconstruct it from labels or pass it?
                # Actually, the batch only has the tokens. Let's pass the text in the batch.
                good_texts = batch['good_text']
                bad_texts = batch['bad_text']
                
                for i in range(len(good_log_prob)):
                    # Re-tokenize with target sweep tokenizer to get T (number of tokens)
                    t1 = len(tok.encode(good_texts[i])) - 1 # Logic from evaluate_blimp: len(tokens[:-1])
                    t2 = len(tok.encode(bad_texts[i])) - 1
                    
                    # 1. Raw Log Prob
                    if good_log_prob[i] > bad_log_prob[i]:
                        sweep_metrics[size]["raw_log_prob"]["correct"] += 1
                    sweep_metrics[size]["raw_log_prob"]["total"] += 1
                    
                    # 2. Per-token Normalized
                    if (good_log_prob[i] / t1) > (bad_log_prob[i] / t2):
                        sweep_metrics[size]["normalized_log_prob"]["correct"] += 1
                    sweep_metrics[size]["normalized_log_prob"]["total"] += 1
                    
                    # 3. BPC (using official bits logic)
                    good_bpc = -good_log_prob[i] / (math.log(2) * good_chars[i])
                    bad_bpc = -bad_log_prob[i] / (math.log(2) * bad_chars[i])
                    if good_bpc < bad_bpc:
                        sweep_metrics[size]["bpc"]["correct"] += 1
                    sweep_metrics[size]["bpc"]["total"] += 1
                    
                    # 4. BPB
                    good_bpb = -good_log_prob[i] / (math.log(2) * good_bytes[i])
                    bad_bpb = -bad_log_prob[i] / (math.log(2) * bad_bytes[i])
                    if good_bpb < bad_bpb:
                        sweep_metrics[size]["bpb"]["correct"] += 1
                    sweep_metrics[size]["bpb"]["total"] += 1

    final_results = {}
    for size in sweep_tokenizers.keys():
        final_results[size] = {
            m: sweep_metrics[size][m]["correct"] / sweep_metrics[size][m]["total"]
            for m in ["raw_log_prob", "normalized_log_prob", "bpc", "bpb"]
        }
    return final_results

def tokenize_decoder_with_text(examples, tokenizer):
    res = tokenize_decoder(examples, tokenizer)
    res['good_text'] = examples['sentence_good']
    res['bad_text'] = examples['sentence_bad']
    return res

def sweep_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['morph', 'standard'], required=True)
    parser.add_argument('--full', action='store_true', help="Run on all 67 subsets")
    args = parser.parse_args()

    # Load Anchor Model
    print(f"Loading Model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)
    model.eval()

    # Anchor Tokenizer (matching evaluate_blimp expectation)
    anchor_spm = os.path.join(args.model_path, "spm.model")
    anchor_tokenizer = DebertaV2Tokenizer(vocab_file=anchor_spm)

    # Sweep Tokenizers
    if args.mode == 'morph':
        base_dir = "data/tokenizers_morph"
        sizes = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000]
        token_paths = {s: os.path.join(base_dir, str(s), "tokenizer.model") for s in sizes if s != 10000}
        token_paths[10000] = anchor_spm
    else:
        base_dir = "data/tokenizers"
        sizes = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000]
        token_paths = {s: os.path.join(base_dir, str(s), "tokenizer.model") for s in sizes if s != 40000}
        token_paths[40000] = anchor_spm

    sweep_tokenizers = {size: DebertaV2Tokenizer(vocab_file=path) for size, path in token_paths.items()}

    BLIMP_SUBSETS = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
    if not args.full:
        BLIMP_SUBSETS = BLIMP_SUBSETS[:10]

    all_results = []

    print(f"Beginning sweep over {len(sweep_tokenizers)} tokenizers...")
    for subset in tqdm(BLIMP_SUBSETS, desc="Subsets"):
        dataset = datasets.load_dataset('json', data_files={'train': os.path.join(args.data_path, f'{subset}.jsonl')})
        tokenize_fn = partial(tokenize_decoder_with_text, tokenizer=anchor_tokenizer)
        dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset['train'].column_names)
        
        # We need a custom collate that keeps text for the sweep re-tokenization
        def collate_with_text(batch):
            text_data = {
                'good_text': [b['good_text'] for b in batch],
                'bad_text': [b['bad_text'] for b in batch]
            }
            padded = padding_collate_fn([{k:v for k,v in b.items() if k not in ['good_text', 'bad_text']} for b in batch])
            padded.update(text_data)
            return padded

        dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=32, shuffle=False, collate_fn=collate_with_text)
        
        subset_res = evaluate_decoder_sweep(model, dataloader, sweep_tokenizers)
        for size, metrics in subset_res.items():
            all_results.append({
                "subset": subset,
                "vocab_size": size,
                **metrics
            })

    df = pd.DataFrame(all_results)
    avg_df = df.groupby("vocab_size").mean(numeric_only=True).reset_index()
    avg_df = avg_df.sort_values("vocab_size")
    
    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Use categorical indexing for even spacing
    x_indices = range(len(avg_df))
    plt.plot(x_indices, avg_df["raw_log_prob"], 'o-', label="Sum (Raw LogProb)", alpha=0.7)
    plt.plot(x_indices, avg_df["normalized_log_prob"], 'D-', label="Per-token Normalized", linewidth=2.5)
    plt.plot(x_indices, avg_df["bpc"], 's--', label="BPC", alpha=0.8)
    plt.plot(x_indices, avg_df["bpb"], 'x-', label="BPB", linewidth=1.5)
    
    # Set X-axis ticks exactly at the indices with labels
    vocab_labels = [f"{int(x/1000)}k" if x >= 1000 else str(int(x)) for x in avg_df["vocab_size"]]
    plt.xticks(x_indices, vocab_labels)
    
    plt.title(f'Stability Sweep ({args.mode.capitalize()}): Aligned Metrics', fontsize=16)
    plt.xlabel('Vocabulary Size', fontsize=12)
    plt.ylabel('BLiMP Accuracy', fontsize=12)
    plt.legend()
    plt.savefig(f'stability_{args.mode}_aligned_sweep.png', dpi=300)
    
    print(f"\n### {args.mode.capitalize()} Aligned Sweep Results")
    print(avg_df.to_string(index=False))
    avg_df.to_json(f"stability_{args.mode}_aligned_results.json", orient='records', indent=2)

if __name__ == "__main__":
    sweep_main()
