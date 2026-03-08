"""
Pairwise Contrastive Fine-Tuning Script

Fine-tunes a pre-trained morphology-aware GPT-2 model using minimal pairs
with margin ranking loss to improve grammatical understanding.

Goal: Improve BLiMP score from 61.81% to 63-65%+

Usage:
    python core/train_pairwise_contrastive.py \
        --model_path outputs/morphology_clean_tokenizer \
        --pairs_file data/minimal_pairs.jsonl \
        --output_dir outputs/morphology_contrastive_finetuned \
        --epochs 3 \
        --margin 1.0 \
        --lambda_margin 0.3
"""

import argparse
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import (
    GPT2LMHeadModel,
    TrainingArguments,
    set_seed
)
import sentencepiece as spm

# Import existing margin ranking trainer
import sys
sys.path.append(str(Path(__file__).parent))
from margin_ranking_trainer import MarginRankingTrainer, PairedDataset, collate_paired_batch


class SentencePieceTokenizerWrapper:
    """Wrapper to make SentencePiece compatible with HuggingFace Trainer."""
    
    def __init__(self, model_path: str):
        """Load SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        self.model_path = model_path  # Store for save_pretrained
        
        # Set special tokens
        self.pad_token_id = self.sp.pad_id()
        self.eos_token_id = self.sp.eos_id()
        self.bos_token_id = self.sp.bos_id()
        self.unk_token_id = self.sp.unk_id()
        
        self.vocab_size = self.sp.vocab_size()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(ids)
    
    def save_pretrained(self, save_directory):
        """Save tokenizer to directory (for HuggingFace Trainer compatibility)."""
        import shutil
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Copy the SentencePiece model file
        src_path = Path(self.model_path)
        if src_path.exists():
            shutil.copy(src_path, save_directory / 'spm.model')
            
            # Also copy vocab if exists
            vocab_src = src_path.parent / 'spm.vocab'
            if vocab_src.exists():
                shutil.copy(vocab_src, save_directory / 'spm.vocab')
    
    def __call__(self, text, truncation=True, max_length=256, padding='max_length', return_tensors='pt'):
        """Tokenize text (HuggingFace-style)."""
        if isinstance(text, str):
            text = [text]
        
        # Encode all texts
        encoded = []
        for t in text:
            ids = self.encode(t, add_special_tokens=True)
            
            # Truncate
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            
            # Pad
            if padding == 'max_length':
                if len(ids) < max_length:
                    ids = ids + [self.pad_token_id] * (max_length - len(ids))
            
            encoded.append(ids)
        
        # Convert to tensors
        if return_tensors == 'pt':
            input_ids = torch.tensor(encoded, dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            return {'input_ids': encoded}


def load_minimal_pairs(pairs_file: str) -> List[Dict]:
    """Load minimal pairs from JSONL or JSON file."""
    pairs_file = Path(pairs_file)
    
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    
    # Try JSONL first
    if pairs_file.suffix == '.jsonl':
        with jsonlines.open(pairs_file, 'r') as reader:
            pairs = list(reader)
    else:  # JSON
        with open(pairs_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
    
    # Normalize format: ensure 'good_sentence' and 'bad_sentence' keys
    normalized = []
    for pair in pairs:
        if 'correct' in pair and 'incorrect' in pair:
            normalized.append({
                'good_sentence': pair['correct'],
                'bad_sentence': pair['incorrect'],
                'category': pair.get('category', 'unknown')
            })
        elif 'good_sentence' in pair and 'bad_sentence' in pair:
            normalized.append(pair)
        else:
            print(f"Warning: Skipping malformed pair: {pair}")
    
    print(f"Loaded {len(normalized):,} minimal pairs from {pairs_file}")
    return normalized


def load_natural_corpus(corpus_path: Optional[str], max_lines: int = 10000) -> List[str]:
    """Load natural sentences for mixed training."""
    if not corpus_path or not Path(corpus_path).exists():
        print("Warning: No natural corpus provided or file not found. Using pairs only.")
        return []
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if len(line.strip()) > 10]
    
    # Sample if too many
    if len(lines) > max_lines:
        import random
        random.shuffle(lines)
        lines = lines[:max_lines]
    
    print(f"Loaded {len(lines):,} natural sentences from {corpus_path}")
    return lines


def print_training_info(args, pairs, natural_sentences):
    """Print training configuration."""
    print("\n" + "="*70)
    print("PAIRWISE CONTRASTIVE FINE-TUNING")
    print("="*70)
    print(f"\nModel: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Pairs file: {args.pairs_file}")
    print(f"Output directory: {args.output_dir}")
    
    print(f"\nDataset:")
    print(f"  Minimal pairs: {len(pairs):,}")
    print(f"  Natural sentences: {len(natural_sentences):,}")
    print(f"  Total training examples: {len(pairs) + len(natural_sentences):,}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Margin: {args.margin}")
    print(f"  Lambda (margin weight): {args.lambda_margin}")
    print(f"  Max length: {args.max_length}")
    print(f"  Device: {args.device}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Pairwise contrastive fine-tuning')
    
    # Model and data
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to SentencePiece tokenizer (default: model_path/spm.model)')
    parser.add_argument('--pairs_file', type=str, required=True,
                        help='Path to minimal pairs file (JSONL or JSON)')
    parser.add_argument('--natural_corpus', type=str, default=None,
                        help='Optional: Path to natural corpus for mixed training')
    parser.add_argument('--max_natural_sentences', type=int, default=10000,
                        help='Maximum natural sentences to include')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for fine-tuned model')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for ranking loss')
    parser.add_argument('--lambda_margin', type=float, default=0.3,
                        help='Weight for margin loss (0.0-1.0)')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, mps, cpu, or auto)')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Log every N steps')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        args.device = device
    
    # Set tokenizer path
    if args.tokenizer_path is None:
        args.tokenizer_path = Path(args.model_path) / 'spm.model'
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = SentencePieceTokenizerWrapper(args.tokenizer_path)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(args.device)
    
    # Load minimal pairs
    pairs = load_minimal_pairs(args.pairs_file)
    
    # Load natural corpus (optional)
    natural_sentences = load_natural_corpus(args.natural_corpus, args.max_natural_sentences)
    
    # Print info
    print_training_info(args, pairs, natural_sentences)
    
    # Create dataset
    print("Creating paired dataset...")
    dataset = PairedDataset(
        natural_examples=natural_sentences,
        synthetic_pairs=pairs,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    print(f"  Total examples: {len(dataset):,}")
    print(f"  Paired examples: {len(pairs):,} ({100*len(pairs)/len(dataset):.1f}%)")
    print(f"  Natural examples: {len(natural_sentences):,} ({100*len(natural_sentences)/len(dataset):.1f}%)")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        fp16=args.device == 'cuda',  # Use FP16 only on CUDA
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=[]  # Disable wandb/tensorboard unless configured
    )
    
    # Create trainer
    print("\nInitializing MarginRankingTrainer...")
    trainer = MarginRankingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collate_paired_batch,
        lambda_margin=args.lambda_margin,
        margin=args.margin
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*70)
    print("SAVING FINAL MODEL")
    print("="*70)
    
    final_output_dir = Path(args.output_dir) / 'final_model'
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(final_output_dir)
    
    # Copy tokenizer files
    import shutil
    tokenizer_src = Path(args.tokenizer_path)
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, final_output_dir / 'spm.model')
        # Also copy vocab if exists
        vocab_src = tokenizer_src.parent / 'spm.vocab'
        if vocab_src.exists():
            shutil.copy(vocab_src, final_output_dir / 'spm.vocab')
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model saved to: {final_output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate on BLiMP:")
    print(f"     python core/evaluate_blimp.py \\")
    print(f"       --model_type decoder \\")
    print(f"       --model_path {final_output_dir} \\")
    print(f"       --tokenizer_path {final_output_dir}/spm.model")
    print()


if __name__ == '__main__':
    main()
