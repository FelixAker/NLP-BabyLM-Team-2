import sentencepiece as spm
import os
import argparse

def train_tokenizer(input_file, vocab_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'tokenizer')
    
    # Define special tokens
    user_defined_symbols = ['[MASK]']
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='unigram',
        user_defined_symbols=user_defined_symbols,
        pad_id=0,
        unk_id=3,
        bos_id=1,
        eos_id=2,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        # Match other potential defaults
        character_coverage=1.0,
        byte_fallback=True
    )
    print(f"Trained tokenizer with vocab_size={vocab_size} in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train.txt')
    parser.add_argument('--sizes', type=int, nargs='+', default=[5000, 10000, 15000, 20000, 25000, 30000])
    parser.add_argument('--output_root', type=str, default='data/tokenizers')
    args = parser.parse_args()
    
    for size in args.sizes:
        output_dir = os.path.join(args.output_root, str(size))
        train_tokenizer(args.input, size, output_dir)
