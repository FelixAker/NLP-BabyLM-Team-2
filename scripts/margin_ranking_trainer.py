"""
E4: Margin Ranking Loss Trainer for Minimal Pairs

Implements contrastive learning with explicit margin ranking:
- L_margin = max(0, margin - (P(good) - P(bad)))
- L_total = L_ce + λ * L_margin

This allows the model to learn from negative evidence (bad sentences)
without accidentally teaching it to be ungrammatical.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class MarginRankingTrainer(Trainer):
    """
    Custom Trainer that implements margin ranking loss for minimal pairs.
    
    Args:
        lambda_margin: Weight for margin loss term (default: 0.3)
        margin: Minimum score difference between good and bad (default: 0.5)
    """
    
    def __init__(self, *args, lambda_margin: float = 0.3, margin: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_margin = lambda_margin
        self.margin = margin
        self.margin_loss_history = []
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined CE + margin ranking loss.
        
        For natural sentences or synthetic good (unpaired): Standard CE loss
        For synthetic pairs: CE loss on good + margin ranking loss
        """
        # Check if this is a paired batch (has both good and bad)
        if 'pair_good_ids' in inputs and 'pair_bad_ids' in inputs:
            return self._compute_paired_loss(model, inputs, return_outputs)
        else:
            # Standard CE loss for natural sentences
            return self._compute_standard_loss(model, inputs, return_outputs)
    
    def _compute_standard_loss(self, model, inputs, return_outputs):
        """Standard cross-entropy loss for natural sentences."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # CE loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_paired_loss(self, model, inputs, return_outputs):
        """
        Compute CE + margin ranking loss for paired examples.
        
        Loss components:
        1. CE loss on good sentence (standard LM objective)
        2. Margin ranking loss: max(0, margin - (score_good - score_bad))
        """
        # Extract paired inputs
        good_ids = inputs.pop('pair_good_ids')
        bad_ids = inputs.pop('pair_bad_ids')
        good_labels = inputs.pop('pair_good_labels')
        bad_labels = inputs.pop('pair_bad_labels')
        
        # 1. Compute CE loss on good sentence
        good_outputs = model(input_ids=good_ids, labels=good_labels)
        loss_ce = good_outputs.loss
        
        # 2. Compute sentence scores for margin ranking
        with torch.no_grad():
            score_good = self._compute_sentence_score(model, good_ids, good_labels)
            score_bad = self._compute_sentence_score(model, bad_ids, bad_labels)
        
        # 3. Margin ranking loss: good should score higher than bad by margin
        # loss = max(0, margin - (score_good - score_bad))
        margin_tensor = torch.tensor(self.margin, device=score_good.device)
        loss_margin = torch.relu(margin_tensor - (score_good - score_bad)).mean()
        
        # 4. Combined loss
        total_loss = loss_ce + self.lambda_margin * loss_margin
        
        # Track margin loss for monitoring
        self.margin_loss_history.append(loss_margin.item())
        
        # Log occasionally
        if len(self.margin_loss_history) % 100 == 0:
            avg_margin_loss = sum(self.margin_loss_history[-100:]) / 100
            print(f"  Margin loss (last 100): {avg_margin_loss:.4f}")
        
        return (total_loss, good_outputs) if return_outputs else total_loss
    
    def _compute_sentence_score(self, model, input_ids, labels):
        """
        Compute sentence-level score as sum of log probabilities.
        
        Higher score = model assigns higher probability to the sentence.
        """
        outputs = model(input_ids=input_ids)
        # Shape: [batch, seq_len, vocab_size]
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        # Align labels for shift (Causal LM)
        # We want to predict token at index t+1 using log_prob at index t
        shift_log_probs = log_probs[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Vectorized gather: pull the log_prob of the actual token
        # Shape: [batch, seq_len-1]
        token_log_probs = torch.gather(shift_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask out padding tokens so they don't affect the score
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        sentence_scores = (token_log_probs * mask).sum(dim=-1)

        return sentence_scores  # Now a Torch tensor, fully on GPU



class PairedDataset(Dataset):
    """
    Dataset that yields paired examples for margin ranking.
    
    Handles three types of examples:
    1. Natural sentences (unpaired)
    2. Synthetic good sentences (unpaired, for exposure)
    3. Synthetic pairs (good + bad, for contrastive learning)
    """
    
    def __init__(self, natural_examples, synthetic_pairs, tokenizer, max_length=256):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add natural examples
        for text in natural_examples:
            self.examples.append({
                'type': 'natural',
                'text': text
            })
        
        # Add synthetic pairs
        for pair in synthetic_pairs:
            self.examples.append({
                'type': 'synthetic_pair',
                'good_text': pair['good_sentence'],
                'bad_text': pair['bad_sentence'],
                'category': pair.get('category', 'unknown')
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        if example['type'] == 'natural':
            # Standard tokenization
            encoding = self.tokenizer(
                example['text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze()
            }
        
        elif example['type'] == 'synthetic_pair':
            # Tokenize both good and bad
            good_enc = self.tokenizer(
                example['good_text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            bad_enc = self.tokenizer(
                example['bad_text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'pair_good_ids': good_enc['input_ids'].squeeze(),
                'pair_bad_ids': bad_enc['input_ids'].squeeze(),
                'pair_good_labels': good_enc['input_ids'].squeeze(),
                'pair_bad_labels': bad_enc['input_ids'].squeeze()
            }


def collate_paired_batch(batch):
    """
    FIXED: Custom collate function for paired batches.
    
    Ensures paired examples are ALWAYS processed when present.
    Keeps paired and unpaired batches separate - never mix them.
    """
    # Separate paired and unpaired
    paired_items = [item for item in batch if 'pair_good_ids' in item]
    unpaired_items = [item for item in batch if 'input_ids' in item]
    
    # CRITICAL FIX: Always prioritize paired items if ANY exist
    # This ensures margin ranking loss actually gets trained
    if paired_items:
        # Use ALL paired items in this batch for margin ranking
        return {
            'pair_good_ids': torch.stack([item['pair_good_ids'] for item in paired_items]),
            'pair_bad_ids': torch.stack([item['pair_bad_ids'] for item in paired_items]),
            'pair_good_labels': torch.stack([item['pair_good_labels'] for item in paired_items]),
            'pair_bad_labels': torch.stack([item['pair_bad_labels'] for item in paired_items])
        }
    else:
        # Only use unpaired if NO paired items exist
        return {
            'input_ids': torch.stack([item['input_ids'] for item in unpaired_items]),
            'attention_mask': torch.stack([item['attention_mask'] for item in unpaired_items]),
            'labels': torch.stack([item['labels'] for item in unpaired_items])
        }
