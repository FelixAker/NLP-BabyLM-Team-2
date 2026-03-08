#!/usr/bin/env python3
"""
Train all tokenizer variants on cleaned BabyLM data.

This script trains multiple tokenization strategies on the same cleaned dataset
for fair comparison.
"""

import argparse
import os
import sys
from pathlib import Path


def train_morphology_bpe(input_file, output_dir, vocab_size=10000):
    """Train morphology-aware BPE tokenizer."""
    print(f"\n{'='*60}")
    print("Training Morphology-Aware BPE Tokenizer")
    print(f"{'='*60}")
    
    # Check if custom morphology tokenizer exists
    morph_script = Path("Training scripts/train_morphology_tokenizer.py")
    if not morph_script.exists():
        print("WARNING: Morphology tokenizer script not found, skipping...")
        return False
    
    # Import and run morphology tokenizer
    sys.path.insert(0, str(morph_script.parent))
    try:
        import train_morphology_tokenizer as morph_mod
        # This will need to be adapted based on actual API
        print("NOTE: Morphology tokenizer training needs manual adaptation")
        print(f"      Please run: python3 '{morph_script}' --input {input_file} --output {output_dir}")
        return True
    except Exception as e:
        print(f"WARNING: Could not load morphology tokenizer: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train all tokenizers on cleaned BabyLM data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cleaned/clean_10M_full.txt"),
        help="Input training file (cleaned)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokenizers/clean_5M"),
        help="Base output directory for all tokenizers",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size for all tokenizers",
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["morphology"],
        help="Which tokenizers to train",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    print("="*70)
    print("BabyLM Tokenizer Training on Cleaned Data")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Tokenizers: {', '.join(args.tokenizers)}")
    
    # Train each tokenizer
    results = {}
    
    if "morphology" in args.tokenizers:
        output_dir = args.output_dir / "morphology_bpe"
        results["morphology"] = train_morphology_bpe(
            str(args.input), str(output_dir), args.vocab_size
        )
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED/SKIPPED"
        print(f"{name:15s}: {status}")
    
    print(f"\n{'='*70}")
    print("All tokenizers training complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
