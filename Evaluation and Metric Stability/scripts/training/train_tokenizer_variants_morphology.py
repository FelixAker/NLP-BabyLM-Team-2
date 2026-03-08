import sentencepiece as spm
import os
import argparse
import json
import re

def get_morphology_rules(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Sort by length descending to match longest first
    prefixes = sorted(config.get('prefixes', []), key=len, reverse=True)
    suffixes = sorted(config.get('suffixes', []), key=len, reverse=True)
    marker = config.get('morpheme_marker', '@@')
    return prefixes, suffixes, marker

def segment_word(word, prefixes, suffixes):
    """
    Splits a word into prefix, stem, and suffix.
    Note: In this implementation, we split into TWO parts if multiple possible, 
    matching the logic of most morphology-aware BPEs.
    We prioritize longest prefix, then longest suffix on the residue.
    """
    prefix = ""
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p):
            prefix = p
            word = word[len(p):]
            break
            
    suffix = ""
    for s in suffixes:
        if word.endswith(s) and len(word) > len(s):
            suffix = s
            word = word[:-len(s)]
            break
            
    parts = []
    if prefix: parts.append(prefix)
    if word: parts.append(word)
    if suffix: parts.append(suffix)
    return parts

def pre_segment_corpus(input_path, output_path, prefixes, suffixes, marker):
    print(f"Pre-segmenting {input_path} to {output_path}...")
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            words = line.split()
            segmented_line = []
            for word in words:
                # Keep special characters/tokens as is
                if word.startswith('*') or word.startswith('[') or word == '...':
                    segmented_line.append(word)
                    continue
                
                # Strip punctuation for segmentation
                clean_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
                if not clean_word:
                    segmented_line.append(word)
                    continue
                    
                parts = segment_word(clean_word.lower(), prefixes, suffixes)
                # Re-attach markers? 
                # If we use SentencePiece BPE, we should join them with spaces 
                # but tell SentencePiece to treat the whole line as a sequence of atoms.
                # Actually, the best way for 'morphology aware' is to join with a marker.
                # segmented_word = marker.join(parts)
                # But our analysis of the vocab showed NO marker.
                # This suggests they were treated as separate 'words'.
                segmented_line.extend(parts)
            
            f_out.write(" ".join(segmented_line) + "\n")

def train_tokenizer(input_file, vocab_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'tokenizer')
    
    # Define special tokens
    user_defined_symbols = ['[MASK]']
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        user_defined_symbols=user_defined_symbols,
        pad_id=0,
        unk_id=3,
        bos_id=1,
        eos_id=2,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        character_coverage=1.0,
        byte_fallback=True
    )
    print(f"Trained morphology-aware BPE tokenizer with vocab_size={vocab_size} in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train.txt')
    parser.add_argument('--config', type=str, default='morphology_bpe/config.json')
    parser.add_argument('--sizes', type=int, nargs='+', default=[5000, 10000, 30000])
    parser.add_argument('--output_root', type=str, default='data/tokenizers_morph')
    args = parser.parse_args()
    
    prefixes, suffixes, marker = get_morphology_rules(args.config)
    
    temp_segmented = 'data/train_morph_segmented.txt'
    pre_segment_corpus(args.input, temp_segmented, prefixes, suffixes, marker)
    
    for size in args.sizes:
        output_dir = os.path.join(args.output_root, str(size))
        train_tokenizer(temp_segmented, size, output_dir)
