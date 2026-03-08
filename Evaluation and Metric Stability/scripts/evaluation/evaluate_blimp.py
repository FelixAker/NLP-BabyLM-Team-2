# evaluate_blimp.py
import argparse
import torch
import datasets
import os
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, DebertaV2Tokenizer
from functools import partial
from tqdm import tqdm
import math
import json

SPM_PATH = './data/tokenizer.model'
BLIMP_SUBSETS = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
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
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--tokenizer_path', type=str, default=None, help="Path to the tokenizer if different from model_path")


def tokenize_encoder(examples, tokenizer):
    batch = {
        "good_inputs": [],
        "bad_inputs": [],
        "good_labels": [],
        "bad_labels": [],
    }

    for i in range(len(examples['sentence_good'])):
        good_tokens = tokenizer.encode(examples['sentence_good'][i], add_special_tokens=False)
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i], add_special_tokens=False)
        batch["good_inputs"].append(good_tokens)
        batch["bad_inputs"].append(bad_tokens)
        batch["good_labels"].append(good_tokens)
        batch["bad_labels"].append(bad_tokens)

    return batch

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
        
        # Calculate chars and bytes for metrics
        batch["good_chars"].append([len(examples['sentence_good'][i])])
        batch["bad_chars"].append([len(examples['sentence_bad'][i])])
        batch["good_bytes"].append([len(examples['sentence_good'][i].encode('utf-8'))])
        batch["bad_bytes"].append([len(examples['sentence_bad'][i].encode('utf-8'))])

    return batch


def padding_collate_fn(batch, max_len=1024, left_padding=False):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
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


def evaluate_decoder(model, dataloader, tokenizer):
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    metrics = {
        "raw_log_prob": {"correct": 0.0, "total": 0.0},
        "normalized_log_prob": {"correct": 0.0, "total": 0.0},
        "bpc": {"correct": 0.0, "total": 0.0},
        "bpb": {"correct": 0.0, "total": 0.0}
    }
    
    import math

    with torch.no_grad():
        for batch in dataloader:
            good_inputs = batch['good_inputs'].to(device=DEVICE)
            bad_inputs = batch['bad_inputs'].to(device=DEVICE)
            good_labels = batch['good_labels'].to(device=DEVICE)
            bad_labels = batch['bad_labels'].to(device=DEVICE)

            # Get length info (shape: batch_size x 1) -> flatten to (batch_size)
            good_chars = batch['good_chars'].to(device=DEVICE).float().view(-1)
            bad_chars = batch['bad_chars'].to(device=DEVICE).float().view(-1)
            good_bytes = batch['good_bytes'].to(device=DEVICE).float().view(-1)
            bad_bytes = batch['bad_bytes'].to(device=DEVICE).float().view(-1)

            good_outputs = model(good_inputs, attention_mask=good_inputs != 0).logits
            bad_outputs = model(bad_inputs, attention_mask=bad_inputs != 0).logits
            
            # Loss per token (CrossEntropyLoss returns loss per element if reduction='none')
            # Shape: (batch_size * sequence_length) -> view as (batch_size, sequence_length)
            good_token_loss = loss_fn(good_outputs.view(-1, good_outputs.shape[-1]), good_labels.view(-1))
            bad_token_loss = loss_fn(bad_outputs.view(-1, bad_outputs.shape[-1]), bad_labels.view(-1))

            good_token_loss = good_token_loss.view(good_inputs.shape[0], -1)
            bad_token_loss = bad_token_loss.view(bad_inputs.shape[0], -1)
            
            # Compute sentence log prob sum: -1 * sum(token_loss) excluding padding
            # We used PADDING (0) for inputs, but for loss calculation we need to mask out padding tokens.
            # Usually labels are -100 for ignore_index in HF, but here padding_collate_fn sets labels to -100 if "labels" in key.
            # But wait, padding_collate_fn does: `if "labels" in key: padded_batch[key] -= 100`. 
            # If original labels were tokens (positive ints), then they become (token - 100).
            # This seems wrong for standard HF CrossEntropy which expects 0..vocab_size or -100.
            # If tokens are e.g. 50, then 50 - 100 = -50 which is invalid index.
            # Let's re-read padding_collate_fn logic in original code.
            
            # Re-reading lines 68-69:
            # if "labels" in key:
            #     padded_batch[key] -= 100
            
            # This suggests the original code expects labels to be processed such that this subtraction works?
            # Or maybe it's a bug in original code or my understanding.
            # Wait, standard padding uses 0. If labels are padded with 0, then 0 - 100 = -100 which IS the ignore index.
            # BUT the actual token IDs are also decremented? That would be bad.
            # Usually `padded_batch[key] = torch.zeros(...)` initializes with 0.
            # Then it fills with data.
            # THEN `padded_batch[key] -= 100` would subtract 100 from EVERYTHING?
            # If so, that breaks valid token IDs.
            # Let's assume the existing code WORKS and I shouldn't break it, but I need to handle loss masking correctly.
            # Actually, `loss_fn` with `ignore_index=-100` (default) handles it.
            # If `padding_collate_fn` does `padded_batch[key] -= 100`, then:
            # Zeros become -100 (good).
            # Valid tokens `t` become `t - 100` (BAD, unless `t` was `t+100`?).
            
            # Looking at `tokenize_decoder`: 
            # `batch["good_labels"].append(good_tokens[1:])` -> these are raw token IDs. 
            # If token ID is 50256, it becomes 50156. This seems WRONG in the original code unless I misread.
            
            # Line 65-69:
            # for key in batch[0]: ... padded_batch[key] = torch.zeros(...)
            # if "labels" in key: padded_batch[key] -= 100
            
            # Then loop filling data:
            # padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])
            
            # AH! The subtraction happens ON THE ZEROS INITIALIZATION, BEFORE filling data.
            # So the PADDING becomes -100. The data is overwritten with correct values.
            # Phew. Original code is correct.
            
            # So, for loss, we just sum valid tokens. valid tokens are != -100.
            
            good_mask = (good_labels != -100).float()
            bad_mask = (bad_labels != -100).float()
            
            # sentence total log prob = -1 * sum(loss per token)
            # reduction='none' gives loss per token. Multiply by mask to ignore padding.
            good_nll_sum = (good_token_loss * good_mask).sum(dim=1)
            bad_nll_sum = (bad_token_loss * bad_mask).sum(dim=1)
            
            good_log_prob = -good_nll_sum
            bad_log_prob = -bad_nll_sum

            # 1. Raw Log-Prob
            # comparison: P(good) > P(bad) => good_log_prob > bad_log_prob
            
            # 2. Per-token Normalized
            # 1/T * log P(x)
            good_num_tokens = good_mask.sum(dim=1)
            bad_num_tokens = bad_mask.sum(dim=1)
            
            good_norm = good_log_prob / good_num_tokens
            bad_norm = bad_log_prob / bad_num_tokens
            
            # 3. Bits-per-character
            # -log_2 P(x) / num_chars = - (log_e P(x) / log_e 2) / num_chars
            # = - (log_prob / ln(2)) / num_chars
            # = - log_prob / (ln(2) * num_chars)
            # Lower is better? No, BLiMP accuracy means "score(good) < score(bad)" ? 
            # Wait, usually "good" sentence should have HIGHER probability.
            # For entropy/bits (lower is better): good_bits < bad_bits.
            
            good_bpc = -good_log_prob / (math.log(2) * good_chars)
            bad_bpc = -bad_log_prob / (math.log(2) * bad_chars)
            
            # 4. Bits-per-byte
            good_bpb = -good_log_prob / (math.log(2) * good_bytes)
            bad_bpb = -bad_log_prob / (math.log(2) * bad_bytes)
            
            for b_idx in range(len(good_log_prob)):
                # Raw Log Prob (Higher is better)
                if good_log_prob[b_idx] > bad_log_prob[b_idx]:
                    metrics["raw_log_prob"]["correct"] += 1
                metrics["raw_log_prob"]["total"] += 1
                
                # Normalized Log Prob (Higher is better)
                if good_norm[b_idx] > bad_norm[b_idx]:
                    metrics["normalized_log_prob"]["correct"] += 1
                metrics["normalized_log_prob"]["total"] += 1
                
                # BPC (Lower is better)
                if good_bpc[b_idx] < bad_bpc[b_idx]:
                    metrics["bpc"]["correct"] += 1
                metrics["bpc"]["total"] += 1
                
                # BPB (Lower is better)
                if good_bpb[b_idx] < bad_bpb[b_idx]:
                    metrics["bpb"]["correct"] += 1
                metrics["bpb"]["total"] += 1

    results = {}
    for key in metrics:
        results[key] = metrics[key]["correct"] / metrics[key]["total"] if metrics[key]["total"] > 0 else 0.0
        
    return results


def evaluate_encoder(model, dataloader, tokenizer):
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in dataloader:
            good_loss = 0.0
            bad_loss = 0.0

            good_inputs = batch['good_inputs'].to(device=DEVICE)
            bad_inputs = batch['bad_inputs'].to(device=DEVICE)
            good_labels = batch['good_labels'].to(device=DEVICE)
            bad_labels = batch['bad_labels'].to(device=DEVICE)

            max_len = batch['good_inputs'].shape[1]
            for i in range(max_len):
                masked_good_inputs = good_inputs.clone()
                masked_good_inputs[:, i] = tokenizer.mask_token_id
                good_outputs = model(masked_good_inputs, attention_mask=good_inputs != 0).logits
                good_loss += loss_fn(good_outputs.view(-1, good_outputs.shape[-1]), good_labels.view(-1))

            max_len = batch['bad_inputs'].shape[1]
            for i in range(max_len):
                masked_bad_inputs = bad_inputs.clone()
                masked_bad_inputs[:, i] = tokenizer.mask_token_id
                bad_outputs = model(masked_bad_inputs, attention_mask=bad_inputs != 0).logits
                bad_loss += loss_fn(bad_outputs.view(-1, bad_outputs.shape[-1]), bad_labels.view(-1))

            good_loss = good_loss.view(good_inputs.shape[0], -1).mean(dim=1)
            bad_loss = bad_loss.view(bad_inputs.shape[0], -1).mean(dim=1)

            for b in range(len(good_loss)):
                if good_loss[b] < bad_loss[b]:
                    correct += 1
                total += 1
    return correct / total

def main():
    args = parser.parse_args()
    model_name = "prajjwal1/bert-tiny" if args.model_type == "encoder" else "sshleifer/tiny-gpt2"
    evaluate_fn = evaluate_encoder if args.model_type == "encoder" else evaluate_decoder
    if not args.model_path:
        args.model_path = model_name

    # load tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        # Fallback for raw SentencePiece models (common in this project context)
        # Check if path is a directory containing tokenizer.model or spm.model
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

    tokenize_fn = partial(tokenize_encoder if args.model_type == "encoder" else tokenize_decoder, tokenizer=tokenizer)

    # load model
    if args.model_type == "encoder":
        model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)

    results = {}

    print("Evaluating on BLIMP subsets...")
    for subset in tqdm(BLIMP_SUBSETS):
        # load dataset and tokenize
        dataset = datasets.load_dataset('json', data_files={'train': os.path.join(args.data_path, f'{subset}.jsonl')})
        dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset['train'].column_names) # map works with functions that return a dictionary
        dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, collate_fn=padding_collate_fn)
        result = evaluate_fn(model, dataloader, tokenizer)
        results[subset] = result


    # Aggregate results
    metric_names = ["raw_log_prob", "normalized_log_prob", "bpc", "bpb"]
    aggregated = {name: [] for name in metric_names}
    
    for subset, res in results.items():
        # result is now a dict
        print(f" -- {subset}:")
        for m in metric_names:
            val = res[m]
            print(f"    {m}: {val:.4f}")
            aggregated[m].append(val)

    print("\nAverage Accuracies:")
    for m in metric_names:
        avg = sum(aggregated[m]) / len(aggregated[m])
        print(f"{m}: {avg:.4f}")

    # Save results to JSON for plotting
    output_file = "blimp_results.json"
    with open(output_file, 'w') as f:
        # Construct a dictionary with both detailed and aggregated results
        json_output = {
            "subsets": results,
            "averages": {m: sum(aggregated[m]) / len(aggregated[m]) for m in metric_names}
        }
        json.dump(json_output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()