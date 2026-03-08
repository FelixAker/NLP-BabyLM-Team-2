import os
import argparse
import json
import re
from transformers import DebertaV2Tokenizer

def load_morphology_rules(config_path):
    if not os.path.exists(config_path):
        return [], [], "@@"
    with open(config_path, 'r') as f:
        config = json.load(f)
    prefixes = set(config.get('prefixes', []))
    suffixes = set(config.get('suffixes', []))
    marker = config.get('morpheme_marker', '@@')
    return prefixes, suffixes, marker

def evaluate_tokenizer(tokenizer_path, test_file, morph_rules):
    prefixes, suffixes, marker = morph_rules
    
    # Load tokenizer
    # Handle different possible paths (spm.model vs tokenizer.model)
    if os.path.isdir(tokenizer_path):
        model_path = os.path.join(tokenizer_path, 'tokenizer.model')
        if not os.path.exists(model_path):
            model_path = os.path.join(tokenizer_path, 'spm.model')
    else:
        model_path = tokenizer_path
        
    try:
        tokenizer = DebertaV2Tokenizer(vocab_file=model_path)
    except Exception as e:
        return {"error": str(e)}

    total_chars = 0
    total_tokens = 0
    total_words = 0
    morph_aligned_tokens = 0
    known_morph_units = prefixes.union(suffixes)
    
    # Analyze a portion of the test file
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:5000] # Limit to 5k lines for speed
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        total_chars += len(line)
        words = line.split()
        total_words += len(words)
        
        for word in words:
            tokens = tokenizer.tokenize(word)
            total_tokens += len(tokens)
            
            for token in tokens:
                # Clean token (remove sentencepiece markers like  )
                clean_token = token.replace(' ', '').replace(' ', '')
                if clean_token in known_morph_units:
                    morph_aligned_tokens += 1
                elif clean_token.lower() in known_morph_units:
                    morph_aligned_tokens += 1
                    
    results = {
        "tokenizer": os.path.basename(os.path.dirname(tokenizer_path)) if os.path.isdir(tokenizer_path) else os.path.basename(tokenizer_path),
        "vocab_size": len(tokenizer),
        "compression_ratio": total_tokens / total_chars if total_chars > 0 else 0,
        "tokens_per_1000_chars": (total_tokens / total_chars * 1000) if total_chars > 0 else 0,
        "avg_tokens_per_word": total_tokens / total_words if total_words > 0 else 0,
        "morph_alignment_score": morph_aligned_tokens / total_tokens if total_tokens > 0 else 0
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--test_file', type=str, default='data/dev.txt')
    parser.add_argument('--config', type=str, default='morphology_bpe/config.json')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    morph_rules = load_morphology_rules(args.config)
    metrics = evaluate_tokenizer(args.tokenizer, args.test_file, morph_rules)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    print(json.dumps(metrics, indent=2))
