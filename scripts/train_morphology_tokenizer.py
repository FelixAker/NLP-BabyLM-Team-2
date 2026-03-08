"""
Morphology-Aware BPE Tokenizer

A novel tokenization approach that respects morphological boundaries.
Pre-splits words into morphemes (prefixes, roots, suffixes) before applying BPE.

This should help models learn grammar better by:
- Keeping grammatical markers separate (-ed, -ing, -s)
- Preserving root meanings
- Better generalization across word forms
"""

import os
import json
import sentencepiece as spm
import tempfile
from collections import Counter
import re

# Common English morphemes
PREFIXES = [
    'un', 're', 'in', 'dis', 'en', 'non', 'pre', 'de', 'mis', 'over', 
    'under', 'out', 'sub', 'inter', 'fore', 'anti', 'mid', 'super'
]

SUFFIXES = [
    'ed', 'ing', 'ly', 'er', 'est', 'ness', 'ment', 'ful', 'less', 
    'tion', 'sion', 'able', 'ible', 'al', 'ial', 'y', 'ous', 'ious',
    'ive', 'ize', 'ise', 'en', 'ate', 'ify', 'hood', 'ship', 'dom'
]

# Inflectional suffixes (most important for grammar)
INFLECTIONS = ['s', 'es', 'ed', 'ing', 'er', 'est']


def split_morphologically(word):
    """
    Split a word into morphological components.
    Returns list of morphemes.
    """
    if len(word) <= 3:
        return [word]
    
    parts = []
    remaining = word.lower()
    
    # Check for prefix
    for prefix in sorted(PREFIXES, key=len, reverse=True):
        if remaining.startswith(prefix) and len(remaining) > len(prefix) + 2:
            parts.append(prefix + '@@')  # @@ marks morpheme boundary
            remaining = remaining[len(prefix):]
            break
    
    # Check for suffix
    suffix_found = None
    for suffix in sorted(SUFFIXES + INFLECTIONS, key=len, reverse=True):
        if remaining.endswith(suffix) and len(remaining) > len(suffix) + 2:
            suffix_found = '@@' + suffix
            remaining = remaining[:-len(suffix)]
            break
    
    # Add root
    if remaining:
        parts.append(remaining)
    
    # Add suffix if found
    if suffix_found:
        parts.append(suffix_found)
    
    return parts if parts else [word]


def preprocess_text_morphologically(text_file, output_file):
    """
    Pre-split text into morphological units before SentencePiece training.
    """
    print("Pre-processing text with morphological boundaries...")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    total_words = 0
    
    for line in lines:
        if not line.strip():
            continue
            
        words = line.strip().split()
        total_words += len(words)
        processed_words = []
        
        for word in words:
            # Keep punctuation and non-alphabetic tokens as-is
            if not any(c.isalpha() for c in word):
                processed_words.append(word)
                continue
            
            # Split morphologically
            morphemes = split_morphologically(word)
            # Join with special marker
            processed_words.append(''.join(morphemes))
        
        # Write as line (SentencePiece needs lines, not one big string)
        processed_lines.append(' '.join(processed_words) + '\n')
    
    # Write preprocessed text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print(f"✓ Preprocessed {total_words:,} words into {len(processed_lines):,} lines")


def train_morphology_aware_bpe(train_file, output_dir, vocab_size=32000):
    """
    Train morphology-aware BPE tokenizer.
    """
    print("=" * 70)
    print("Training Morphology-Aware BPE Tokenizer")
    print("=" * 70)
    print(f"Vocab size: {vocab_size:,}")
    print(f"Input: {train_file}")
    print(f"Output: {output_dir}")
    
    # Create temporary file for preprocessed text
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp:
        temp_file = tmp.name
        print(f"\nPreprocessing with morphological splits...")
        preprocess_text_morphologically(train_file, temp_file)
    
    try:
        # Train SentencePiece BPE on morphologically-split text
        model_prefix = os.path.join(output_dir, 'tokenizer')
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nTraining BPE on morphological units...")
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            user_defined_symbols=['@@'],  # Morpheme boundary marker
            byte_fallback=True,
        )
        
        print(f"✓ Trained morphology-aware BPE tokenizer")
        
        # Save config
        config = {
            'type': 'morphology_bpe',
            'vocab_size': vocab_size,
            'morpheme_marker': '@@',
            'prefixes': PREFIXES,
            'suffixes': SUFFIXES + INFLECTIONS,
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Morphology-aware tokenizer saved to: {output_dir}")
        print(f"   Files: tokenizer.model, tokenizer.vocab, config.json")
        
        # Test it
        print("\n" + "=" * 70)
        print("Testing Tokenizer")
        print("=" * 70)
        
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        
        test_sentences = [
            "The dogs are running quickly.",
            "She unhappily walked away.",
            "They reopened the investigation."
        ]
        
        for sent in test_sentences:
            tokens = sp.encode_as_pieces(sent)
            print(f"\nInput:  {sent}")
            print(f"Tokens: {tokens}")
        
        print("\n" + "=" * 70)
        print("✅ Done! Ready to train a model with morphology-aware tokenizer")
        print("=" * 70)
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    train_file = 'data/train.txt'
    output_dir = 'data/tokenizers/morphology_bpe_32k'
    
    train_morphology_aware_bpe(train_file, output_dir, vocab_size=32000)
