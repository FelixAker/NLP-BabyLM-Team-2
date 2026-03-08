#!/usr/bin/env python3
"""
Token Imbalance Analysis for BLiMP Evaluation

This script analyzes the correlation between token count differences (Δtokens)
in BLiMP sentence pairs and model prediction errors.
"""

import argparse
import torch
import datasets
import os
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DebertaV2Tokenizer
from functools import partial
from tqdm import tqdm
import math
from collections import defaultdict
from scipy import stats

BLIMP_SUBSETS = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']

# Categorize subsets
CATEGORIES = {
    'wh-movement': [s for s in BLIMP_SUBSETS if 'wh_' in s or 'wh-' in s],
    'agreement': [s for s in BLIMP_SUBSETS if 'agreement' in s],
    'NPIs': [s for s in BLIMP_SUBSETS if 'npi' in s]
}

if os.environ.get("FORCE_CPU") == "1":
    DEVICE = 'cpu'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True, choices=['encoder', 'decoder'])
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--tokenizer_path', type=str, default=None)
parser.add_argument('--output_file', type=str, default='token_imbalance_results.json')


def tokenize_decoder(examples, tokenizer):
    """Tokenize decoder examples and compute token counts."""
    batch = {
        "good_inputs": [],
        "bad_inputs": [],
        "good_labels": [],
        "bad_labels": [],
        "good_chars": [],
        "bad_chars": [],
        "good_bytes": [],
        "bad_bytes": [],
        "good_token_count": [],
        "bad_token_count": [],
        "sentence_good": [],
        "sentence_bad": [],
    }

    for i in range(len(examples['sentence_good'])):
        good_tokens = tokenizer.encode(examples['sentence_good'][i])
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i])
        
        batch["good_inputs"].append(good_tokens[:-1])
        batch["bad_inputs"].append(bad_tokens[:-1])
        batch["good_labels"].append(good_tokens[1:])
        batch["bad_labels"].append(bad_tokens[1:])
        
        # Calculate chars and bytes for metrics
        batch["good_chars"].append([len(examples['sentence_good'][i])])
        batch["bad_chars"].append([len(examples['sentence_bad'][i])])
        batch["good_bytes"].append([len(examples['sentence_good'][i].encode('utf-8'))])
        batch["bad_bytes"].append([len(examples['sentence_bad'][i].encode('utf-8'))])
        
        # Store token counts (excluding BOS/EOS for fair comparison)
        batch["good_token_count"].append([len(good_tokens) - 1])
        batch["bad_token_count"].append([len(bad_tokens) - 1])
        
        # Store original sentences for reference
        batch["sentence_good"].append([examples['sentence_good'][i]])
        batch["sentence_bad"].append([examples['sentence_bad'][i]])

    return batch


def padding_collate_fn(batch, max_len=1024, left_padding=False):
    """Pads each list with zeros and concatenates by key."""
    padded_batch = {}
    for key in batch[0]:
        if key in ["sentence_good", "sentence_bad"]:
            # Don't pad sentences, just collect them
            padded_batch[key] = [b[key] for b in batch]
            continue
            
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            if key in ["sentence_good", "sentence_bad"]:
                continue
            key_len = min(max_len, len(sample[key]))
            if left_padding:
                padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])
            else:
                padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch


def evaluate_decoder_with_token_counts(model, dataloader, tokenizer):
    """Evaluate decoder model and track token count differences."""
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # Store detailed results for each pair
    pair_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            good_inputs = batch['good_inputs'].to(device=DEVICE)
            bad_inputs = batch['bad_inputs'].to(device=DEVICE)
            good_labels = batch['good_labels'].to(device=DEVICE)
            bad_labels = batch['bad_labels'].to(device=DEVICE)

            # Get length info
            good_chars = batch['good_chars'].to(device=DEVICE).float().view(-1)
            bad_chars = batch['bad_chars'].to(device=DEVICE).float().view(-1)
            good_bytes = batch['good_bytes'].to(device=DEVICE).float().view(-1)
            bad_bytes = batch['bad_bytes'].to(device=DEVICE).float().view(-1)
            good_token_counts = batch['good_token_count'].to(device=DEVICE).float().view(-1)
            bad_token_counts = batch['bad_token_count'].to(device=DEVICE).float().view(-1)

            good_outputs = model(good_inputs, attention_mask=good_inputs != 0).logits
            bad_outputs = model(bad_inputs, attention_mask=bad_inputs != 0).logits
            
            # Loss per token
            good_token_loss = loss_fn(good_outputs.view(-1, good_outputs.shape[-1]), good_labels.view(-1))
            bad_token_loss = loss_fn(bad_outputs.view(-1, bad_outputs.shape[-1]), bad_labels.view(-1))

            good_token_loss = good_token_loss.view(good_inputs.shape[0], -1)
            bad_token_loss = bad_token_loss.view(bad_inputs.shape[0], -1)
            
            good_mask = (good_labels != -100).float()
            bad_mask = (bad_labels != -100).float()
            
            good_nll_sum = (good_token_loss * good_mask).sum(dim=1)
            bad_nll_sum = (bad_token_loss * bad_mask).sum(dim=1)
            
            good_log_prob = -good_nll_sum
            bad_log_prob = -bad_nll_sum

            # Compute metrics
            good_num_tokens = good_mask.sum(dim=1)
            bad_num_tokens = bad_mask.sum(dim=1)
            
            good_norm = good_log_prob / good_num_tokens
            bad_norm = bad_log_prob / bad_num_tokens
            
            good_bpc = -good_log_prob / (math.log(2) * good_chars)
            bad_bpc = -bad_log_prob / (math.log(2) * bad_chars)
            
            good_bpb = -good_log_prob / (math.log(2) * good_bytes)
            bad_bpb = -bad_log_prob / (math.log(2) * bad_bytes)
            
            # Store results for each pair
            for b_idx in range(len(good_log_prob)):
                delta_tokens = int(good_token_counts[b_idx].item() - bad_token_counts[b_idx].item())
                
                pair_result = {
                    'sentence_good': batch['sentence_good'][b_idx][0],
                    'sentence_bad': batch['sentence_bad'][b_idx][0],
                    'tokens_good': int(good_token_counts[b_idx].item()),
                    'tokens_bad': int(bad_token_counts[b_idx].item()),
                    'delta_tokens': delta_tokens,
                    'abs_delta_tokens': abs(delta_tokens),
                    'predictions': {
                        'raw_log_prob': {
                            'correct': bool(good_log_prob[b_idx] > bad_log_prob[b_idx]),
                            'good_score': float(good_log_prob[b_idx].item()),
                            'bad_score': float(bad_log_prob[b_idx].item())
                        },
                        'normalized_log_prob': {
                            'correct': bool(good_norm[b_idx] > bad_norm[b_idx]),
                            'good_score': float(good_norm[b_idx].item()),
                            'bad_score': float(bad_norm[b_idx].item())
                        },
                        'bpc': {
                            'correct': bool(good_bpc[b_idx] < bad_bpc[b_idx]),
                            'good_score': float(good_bpc[b_idx].item()),
                            'bad_score': float(bad_bpc[b_idx].item())
                        },
                        'bpb': {
                            'correct': bool(good_bpb[b_idx] < bad_bpb[b_idx]),
                            'good_score': float(good_bpb[b_idx].item()),
                            'bad_score': float(bad_bpb[b_idx].item())
                        }
                    }
                }
                pair_results.append(pair_result)
        
    return pair_results


def compute_statistics(all_results):
    """Compute correlation statistics and binned analysis."""
    
    metrics = ['raw_log_prob', 'normalized_log_prob', 'bpc', 'bpb']
    
    # Collect data for correlation analysis
    abs_deltas = []
    errors_by_metric = {m: [] for m in metrics}
    
    for result in all_results:
        abs_deltas.append(result['abs_delta_tokens'])
        for metric in metrics:
            # 1 if error, 0 if correct
            error = 0 if result['predictions'][metric]['correct'] else 1
            errors_by_metric[metric].append(error)
    
    abs_deltas = np.array(abs_deltas)
    
    # Compute correlations
    correlations = {}
    for metric in metrics:
        errors = np.array(errors_by_metric[metric])
        corr, p_value = stats.pearsonr(abs_deltas, errors)
        correlations[metric] = {
            'correlation': float(corr),
            'p_value': float(p_value)
        }
    
    # Binned analysis: accuracy by |Δtokens|
    bins = [0, 1, 2, 3, 4, float('inf')]
    bin_labels = ['0', '1', '2', '3', '4+']
    
    binned_stats = {label: {m: {'correct': 0, 'total': 0} for m in metrics} for label in bin_labels}
    
    for result in all_results:
        abs_delta = result['abs_delta_tokens']
        
        # Find bin
        bin_idx = 0
        for i in range(len(bins) - 1):
            if bins[i] <= abs_delta < bins[i + 1]:
                bin_idx = i
                break
        
        bin_label = bin_labels[bin_idx]
        
        for metric in metrics:
            if result['predictions'][metric]['correct']:
                binned_stats[bin_label][metric]['correct'] += 1
            binned_stats[bin_label][metric]['total'] += 1
    
    # Compute accuracies
    binned_accuracies = {}
    for bin_label in bin_labels:
        binned_accuracies[bin_label] = {}
        for metric in metrics:
            total = binned_stats[bin_label][metric]['total']
            if total > 0:
                acc = binned_stats[bin_label][metric]['correct'] / total
                binned_accuracies[bin_label][metric] = {
                    'accuracy': float(acc),
                    'error_rate': float(1 - acc),
                    'count': int(total)
                }
            else:
                binned_accuracies[bin_label][metric] = {
                    'accuracy': 0.0,
                    'error_rate': 0.0,
                    'count': 0
                }
    
    return {
        'correlations': correlations,
        'binned_accuracies': binned_accuracies
    }


def compute_category_statistics(all_results, subset_to_category):
    """Compute statistics broken down by linguistic category."""
    
    metrics = ['raw_log_prob', 'normalized_log_prob', 'bpc', 'bpb']
    
    category_stats = {}
    
    for category_name, subsets in CATEGORIES.items():
        # Filter results for this category
        category_results = [r for r in all_results if r['subset'] in subsets]
        
        if not category_results:
            continue
        
        # Compute correlations for this category
        abs_deltas = []
        errors_by_metric = {m: [] for m in metrics}
        
        for result in category_results:
            abs_deltas.append(result['abs_delta_tokens'])
            for metric in metrics:
                error = 0 if result['predictions'][metric]['correct'] else 1
                errors_by_metric[metric].append(error)
        
        abs_deltas = np.array(abs_deltas)
        
        correlations = {}
        for metric in metrics:
            errors = np.array(errors_by_metric[metric])
            if len(errors) > 1 and np.std(errors) > 0 and np.std(abs_deltas) > 0:
                corr, p_value = stats.pearsonr(abs_deltas, errors)
                correlations[metric] = {
                    'correlation': float(corr),
                    'p_value': float(p_value)
                }
            else:
                correlations[metric] = {
                    'correlation': 0.0,
                    'p_value': 1.0
                }
        
        category_stats[category_name] = {
            'correlations': correlations,
            'sample_size': len(category_results)
        }
    
    return category_stats


def main():
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        if os.path.isdir(tokenizer_path):
            spm_file = os.path.join(tokenizer_path, 'tokenizer.model')
            if not os.path.exists(spm_file):
                spm_file = os.path.join(tokenizer_path, 'spm.model')
            
            if os.path.exists(spm_file):
                print(f"Loading DebertaV2Tokenizer from {spm_file}...")
                tokenizer = DebertaV2Tokenizer(vocab_file=spm_file)
            else:
                raise
        elif os.path.isfile(tokenizer_path) and tokenizer_path.endswith(".model"):
            print(f"Loading DebertaV2Tokenizer from {tokenizer_path}...")
            tokenizer = DebertaV2Tokenizer(vocab_file=tokenizer_path)
        else:
            raise

    tokenize_fn = partial(tokenize_decoder, tokenizer=tokenizer)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)

    all_results = []
    
    print("Analyzing BLiMP subsets with token count tracking...")
    for subset in tqdm(BLIMP_SUBSETS):
        # Load dataset and tokenize
        dataset = datasets.load_dataset('json', data_files={'train': os.path.join(args.data_path, f'{subset}.jsonl')})
        dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset['train'].column_names)
        dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn)
        
        pair_results = evaluate_decoder_with_token_counts(model, dataloader, tokenizer)
        
        # Add subset information
        for result in pair_results:
            result['subset'] = subset
        
        all_results.extend(pair_results)
    
    print(f"\nTotal pairs analyzed: {len(all_results)}")
    
    # Compute overall statistics
    print("Computing correlation statistics...")
    overall_stats = compute_statistics(all_results)
    
    # Compute category-level statistics
    print("Computing category-level statistics...")
    subset_to_category = {}
    for cat_name, subsets in CATEGORIES.items():
        for subset in subsets:
            subset_to_category[subset] = cat_name
    
    category_stats = compute_category_statistics(all_results, subset_to_category)
    
    # Prepare output
    output = {
        'model_path': args.model_path,
        'total_pairs': len(all_results),
        'overall_statistics': overall_stats,
        'category_statistics': category_stats,
        'pairs': all_results  # Full detailed results
    }
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n=== Summary ===")
    print("\nOverall Correlations (|Δtokens| vs Error):")
    for metric, corr_data in overall_stats['correlations'].items():
        print(f"  {metric}: r={corr_data['correlation']:.4f}, p={corr_data['p_value']:.4e}")
    
    print("\nCategory-Level Correlations:")
    for category, cat_data in category_stats.items():
        print(f"\n  {category} (n={cat_data['sample_size']}):")
        for metric, corr_data in cat_data['correlations'].items():
            print(f"    {metric}: r={corr_data['correlation']:.4f}, p={corr_data['p_value']:.4e}")
    
    print(f"\nDetailed results saved to {args.output_file}")


if __name__ == '__main__':
    main()
