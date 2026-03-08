import argparse
import torch
import datasets
import os
import json
import sys
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast
from functools import partial
from tqdm import tqdm

# FIX: Ensure we can import local modules regardless of where we run this script from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing the scorer, handle failure gracefully
try:
    from critical_region_scorer import critical_region_score, critical_region_score_with_details
except ImportError:
    print("WARNING: Could not import 'critical_region_scorer'. 'critical-region' scoring will fail.")

BLIMP_SUBSETS = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Force CUDA on LRZ

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, required=True, choices=['encoder', 'decoder'])
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--scoring_method', type=str, default='full', choices=['full', 'critical-region', 'both'])
parser.add_argument('--output_path', type=str, default="")

def tokenize_decoder(examples, tokenizer):
    batch = {"good_inputs": [], "bad_inputs": [], "good_labels": [], "bad_labels": []}
    for i in range(len(examples['sentence_good'])):
        # Add eos_token (or bos) if your training did so. Standard GPT-2 usually doesn't need explicit BOS for likelihood.
        # But consistency with training is key.
        good_tokens = tokenizer.encode(examples['sentence_good'][i])
        bad_tokens = tokenizer.encode(examples['sentence_bad'][i])
        
        # Standard causal masking: Input is x[0..N-1], Label is x[1..N]
        batch["good_inputs"].append(good_tokens[:-1])
        batch["bad_inputs"].append(bad_tokens[:-1])
        batch["good_labels"].append(good_tokens[1:])
        batch["bad_labels"].append(bad_tokens[1:])
    return batch

def padding_collate_fn(batch, max_len=1024):
    padded_batch = {}
    for key in batch[0]:
        # Determine max length in this batch
        largest = min(max_len, max([len(b[key]) for b in batch]))
        # Create tensor of ones (or zeros) filled with padding ID
        # FIX: Ensure we use the tokenizer's actual pad_token_id. 
        # Since we don't have the tokenizer object here easily, assuming 0 is often safe for custom SP, 
        # but -100 is required for LABELS.
        
        if "labels" in key:
            padded_batch[key] = torch.full((len(batch), largest), -100, dtype=torch.long)
        else:
            # Input padding: 0 is common for SP, but check your tokenizer!
            padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            # Right padding (Standard for GPT-2 generation, but for scoring likelihoods right padding is also fine)
            padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch

def evaluate_decoder(model, dataloader):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    correct = 0.0
    total = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating batch"):
            good_inputs = batch['good_inputs'].to(DEVICE)
            bad_inputs = batch['bad_inputs'].to(DEVICE)
            good_labels = batch['good_labels'].to(DEVICE)
            bad_labels = batch['bad_labels'].to(DEVICE)

            # Calculate Good Sentence Likelihood
            good_logits = model(good_inputs).logits
            # Shift logits/labels is handled by the tokenizer function (inputs=:-1, labels=1:)
            # We calculate loss per token, then average over sentence
            good_loss = loss_fn(good_logits.transpose(1, 2), good_labels)
            # Mask out the -100 padding tokens from the average
            good_mask = (good_labels != -100).float()
            good_sent_score = (good_loss * good_mask).sum(dim=1) / good_mask.sum(dim=1)

            # Calculate Bad Sentence Likelihood
            bad_logits = model(bad_inputs).logits
            bad_loss = loss_fn(bad_logits.transpose(1, 2), bad_labels)
            bad_mask = (bad_labels != -100).float()
            bad_sent_score = (bad_loss * bad_mask).sum(dim=1) / bad_mask.sum(dim=1)

            # Lower loss = Higher probability = "Correct"
            comparisons = good_sent_score < bad_sent_score
            correct += comparisons.sum().item()
            total += good_inputs.size(0)
            
    return correct / total if total > 0 else 0

def main():
    args = parser.parse_args()
    
    # 1. Load Tokenizer (FIXED LOGIC)
    print(f"Loading tokenizer from {args.model_path}")
    
    # Custom SentencePiece loader since AutoTokenizer falls back to fast tokenizer that crashes on plain SP models
    class SentencePieceTokenizer:
        def __init__(self, model_path):
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            self.vocab_size = self.sp.vocab_size()
            self.pad_token_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 0
            self.bos_token_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 1
            self.eos_token_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 2
        def encode(self, text, out_type=int):
            return self.sp.encode(text, out_type=out_type)
        def decode(self, ids):
            return self.sp.decode(ids)

    try:
        # First try loading as a standard AutoTokenizer (handles fast/slow automatically)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"AutoTokenizer failed: {e}")
        # Fallback for raw .model files if AutoTokenizer fails
        possible_spm = os.path.join(args.model_path, 'tokenizer.model')
        possible_spm2 = os.path.join(args.model_path, 'spm.model')
        
        if os.path.exists(possible_spm):
            print(f"Falling back to custom SentencePiece for {possible_spm}")
            tokenizer = SentencePieceTokenizer(possible_spm)
        elif os.path.exists(possible_spm2):
            print(f"Falling back to custom SentencePiece for {possible_spm2}")
            tokenizer = SentencePieceTokenizer(possible_spm2)
        else:
            raise ValueError("Could not load tokenizer. Ensure tokenizer.model, spm.model or config.json exists.")

    # Ensure pad token exists for the collator
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0 

    # 2. Load Model
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(DEVICE)

    # 3. Prepare Data
    tokenize_fn = partial(tokenize_decoder, tokenizer=tokenizer)
    results = {}

    for subset in BLIMP_SUBSETS:
        # Handle file paths robustly
        file_name = f"{subset}.jsonl"
        # Check specific locations: root of data_path, or inside 'data' folder
        possible_paths = [
            os.path.join(args.data_path, file_name),
            os.path.join(args.data_path, 'data', file_name)
        ]
        
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if not found_path:
            print(f"Skipping {subset}: File not found in {args.data_path}")
            continue

        # Load and Process
        raw_dataset = datasets.load_dataset('json', data_files={'train': found_path}, split='train')
        tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=raw_dataset.column_names)
        
        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset, 
            batch_size=args.batch_size, 
            collate_fn=padding_collate_fn
        )

        # Evaluate
        score = evaluate_decoder(model, dataloader)
        results[subset] = score
        print(f"{subset}: {score:.4f}")

    # 4. Save Results
    print(f"Average Score: {sum(results.values())/len(results):.4f}")
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()