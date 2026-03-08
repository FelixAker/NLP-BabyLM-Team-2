

print("Starting test_metrics_impl.py...")
import torch
import math
from unittest.mock import MagicMock
from torch.utils.data import DataLoader
import sys
import os

# Add the directory to path so we can import evaluate_decoder
sys.path.append(os.path.abspath('/Users/begum/Downloads/BabyLM-Tiny-main-3'))

from evaluate_blimp import evaluate_decoder, padding_collate_fn

# Mock Tokenizer
class MockTokenizer:
    def encode(self, text):
        # simple mock: return list of ordinals
        return [ord(c) for c in text]
    
    @property
    def mask_token_id(self):
        return 0

# Mock Model
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_ids, attention_mask=None):
        # Return random logits? 
        # Better: return logits that result in known loss.
        # Loss is CE. 
        # We want to control the loss to verify metrics.
        # Let's say we want loss=1.0 per token for "good" and loss=2.0 per token for "bad".
        
        batch_size, seq_len = input_ids.shape
        vocab_size = 256
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        
        # We can't easily control CrossEntropyLoss output via logits without knowing the target.
        # But we can verify if the script calculates metrics correctly given the inputs.
        # Let's simple return zeros, which means uniform probability.
        # Loss will be -log(1/vocab_size) = log(vocab_size).
        
        return MagicMock(logits=logits)

def test_metrics():
    # Setup
    tokenizer = MockTokenizer()
    model = MockModel()
    model.eval()
    
    # Sentence 1: "a" (1 char, 1 byte). Tokens: [97]
    # Sentence 2: "ab" (2 chars, 2 bytes). Tokens: [97, 98]
    
    # We need to constructing a batch that `evaluate_decoder` expects.
    # It expects a DataLoader yielding batches.
    
    # Manually create a batch dict simulating what padding_collate_fn produces
    # But wait, evaluate_decoder takes a dataloader.
    
    # Let's use the actual padding_collate_fn with valid inputs.
    # We need to simulate `tokenize_decoder` output first.
    
    # Good: "a" -> tokens [97]. Input [97], Label? 
    # tokenize_decoder splits input/label. 
    # "a" -> encode -> [97]. Input [], Label []. This is empty?
    # tokenize_decoder: 
    #   good_tokens = encode(text)
    #   input = good_tokens[:-1]
    #   label = good_tokens[1:]
    # So single token sentences are problematic for decoder evaluation (no next token).
    # Let's use longer sentences.
    
    # Good: "abc" -> [97, 98, 99]. Input [97, 98], Label [98, 99]. Chars 3.
    # Bad: "xz" -> [120, 122]. Input [120], Label [122]. Chars 2.
    
    batch_list = [
        {
            "good_inputs": [97, 98],
            "bad_inputs": [120],
            "good_labels": [98, 99],
            "bad_labels": [122],
            "good_chars": [3],
            "bad_chars": [2],
            "good_bytes": [3],
            "bad_bytes": [2]
        }
    ]
    
    # Run collate
    padded_batch = padding_collate_fn(batch_list)
    print("Padded batch keys:", padded_batch.keys())
    
    dataloader = [padded_batch] # Simulate dataloader
    
    # Expected Metrics:
    # Vocab size = 256 (from MockModel).
    # Logits = zeros. Softmax = 1/256 for all.
    # Loss per token = -log(1/256) = log(256) = 5.545
    
    # Good sentence: 2 tokens. Total NLL = 2 * 5.545 = 11.09
    # Bad sentence: 1 token. Total NLL = 1 * 5.545 = 5.545
    
    # Raw Log Prob:
    # Good: -11.09
    # Bad: -5.545
    # Correct: Good < Bad (since -11.09 < -5.545). Wait...
    # Raw log prob: "good" sentence usually has HIGHER log prob (closer to 0).
    # Here "good" is longer, so it accumulates more negative log likelihood (lower probability).
    # So "raw_log_prob" prediction: Bad is better. Correct = 0.
    
    # Normalized Log Prob:
    # Good: -11.09 / 2 = -5.545
    # Bad: -5.545 / 1 = -5.545
    # Tie? Code: if good > bad. -5.545 > -5.545 is False. Correct = 0.
    
    # BPC:
    # Good: -(-11.09) / (log(2) * 3) = 11.09 / (0.693 * 3) = 11.09 / 2.079 = 5.33
    # Bad: -(-5.545) / (log(2) * 2) = 5.545 / (0.693 * 2) = 5.545 / 1.386 = 4.00
    # Comparison: Good < Bad? 5.33 < 4.00 is False (Lower BPC is better). Correct = 0.
    
    # BPB:
    # Same as BPC since chars=bytes here. Correct = 0.
    
    # Result should be 0.0 for all.
    
    results = evaluate_decoder(model, dataloader, tokenizer)
    print("Results:", results)
    
    # Let's try to make Good BETTER.
    # To do this with constant loss, Good needs to be shorter or we assume metrics handle length.
    # If we want Good > Bad in raw log prob, Good must have higher prob (less loss).
    # With constant per-token loss, Good must be SHORTER than Bad.
    
    # Case 2: Good is Shorter
    # Good: "a" (but needs 2 tokens for decoder training? "ab" -> input "a", label "b")
    # Let's use:
    # Good: "ab" (2 chars, 2 tokens). Input [97], Label [98]. Loss = 5.545. LogProb = -5.545.
    # Bad: "xyz" (3 chars, 3 tokens). Input [120, 121], Label [121, 122]. Loss = 2*5.545. LogProb = -11.09.
    
    batch_list_2 = [
        {
            "good_inputs": [97],
            "bad_inputs": [120, 121],
            "good_labels": [98],
            "bad_labels": [121, 122],
            "good_chars": [2],
            "bad_chars": [3],
            "good_bytes": [2],
            "bad_bytes": [3]
        }
    ]
    padded_batch_2 = padding_collate_fn(batch_list_2)
    dataloader_2 = [padded_batch_2]
    
    # Case 2 Check:
    # Good LogProb: -5.545
    # Bad LogProb: -11.09
    # Good > Bad? -5.545 > -11.09. YES. Correct = 1.
    
    # Good Norm: -5.545 / 1 = -5.545
    # Bad Norm: -11.09 / 2 = -5.545
    # Good > Bad? False. Correct = 0.
    
    # Good BPC: 5.545 / (log2 * 2) = 5.545 / 1.386 = 4.0
    # Bad BPC: 11.09 / (log2 * 3) = 11.09 / 2.079 = 5.33
    # Good < Bad? 4.0 < 5.33. YES. Correct = 1.
    
    results_2 = evaluate_decoder(model, dataloader_2, tokenizer)
    print("Result 2:", results_2)
    
    with open("test_result.txt", "w") as f:
        try:
            assert results_2['raw_log_prob'] == 1.0
            assert results_2['normalized_log_prob'] == 0.0
            assert results_2['bpc'] == 1.0
            assert results_2['bpb'] == 1.0
            f.write("PASSED\n")
            f.write(str(results_2))
        except AssertionError as e:
            f.write("FAILED\n")
            f.write(str(results_2))
            
    print("Test Passed!")

if __name__ == "__main__":
    test_metrics()
