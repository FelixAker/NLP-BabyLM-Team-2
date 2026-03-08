#!/usr/bin/env python3
"""
GLUE Evaluation for morphology_clean_tokenizer model.

Adapts the standard GLUE evaluation to work with our custom SentencePiece tokenizer
and GPT-2 CLM model by adding a classification head and fine-tuning.
"""

import argparse
import json
import datasets
import torch
import torch.nn as nn
import sentencepiece as spm
from transformers import GPT2LMHeadModel, GPT2Config
from functools import partial
from tqdm import tqdm

# GLUE tasks we'll evaluate
GLUE_TASKS = ['mrpc', 'qqp', 'boolq', 'sst2', 'rte', 'cola']

TASK_INPUTS = {
    'mrpc': ['sentence1', 'sentence2'],
    'qqp': ['question1', 'question2'],
    'boolq': ['question', 'passage'],
    'sst2': ['sentence'],
    'rte': ['sentence1', 'sentence2'],
    'cola': ['sentence'],
}

TASK_NUM_LABELS = {
    'mrpc': 2,
    'qqp': 2,
    'boolq': 2,
    'sst2': 2,
    'rte': 2,
    'cola': 2,
}

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


class SentencePieceTokenizerWrapper:
    """Wrapper to make SentencePiece compatible with our pipeline."""
    
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.vocab_size()
        self.pad_token_id = self.sp.pad_id()
        self.model_max_length = 512
    
    def encode(self, text, truncation=True, max_length=None):
        """Encode text to token IDs."""
        max_len = max_length or self.model_max_length
        ids = self.sp.encode(text, out_type=int)
        if truncation and len(ids) > max_len:
            ids = ids[:max_len]
        return ids
    
    def decode(self, ids):
        """Decode token IDs to text."""
        return self.sp.decode(ids)


class GPT2ForSequenceClassification(nn.Module):
    """
    GPT-2 model with a classification head on top.
    Uses pooling of the last token's hidden state (like GPT-2 original).
    """
    
    def __init__(self, base_model_path, num_labels=2):
        super().__init__()
        
        # Load base GPT-2 model - check for nested directory structure
        import os
        if os.path.exists(f"{base_model_path}/morphology_clean_tokenizer"):
            model_path = f"{base_model_path}/morphology_clean_tokenizer"
        else:
            model_path = base_model_path
        
        self.transformer = GPT2LMHeadModel.from_pretrained(model_path).transformer
        self.config = self.transformer.config
        
        # Classification head
        self.score = nn.Linear(self.config.n_embd, num_labels, bias=False)
        
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) for classification
        
        Returns:
            dict with 'loss' and 'logits'
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_states = transformer_outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # Pool: use last token's hidden state
        # Find the last non-padding token for each sequence
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
        else:
            sequence_lengths = torch.full((input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device)
        
        # Get hidden state of last token
        pooled_hidden_states = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
        
        # Classification
        logits = self.score(pooled_hidden_states)  # (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}


def tokenize_function(examples, tokenizer, task):
    """Tokenize examples for a specific GLUE task."""
    batch = {
        "input_ids": [],
        "labels": [],
    }
    
    input_fields = TASK_INPUTS[task]
    
    for i in range(len(examples['label'])):
        # Concatenate input fields with a separator
        input_texts = [examples[field][i] for field in input_fields]
        input_text = " ".join(input_texts)
        
        # Tokenize
        input_ids = tokenizer.encode(input_text, truncation=True)
        
        batch["input_ids"].append(input_ids)
        batch["labels"].append(examples['label'][i])
    
    return batch


def padding_collate_fn(batch, max_len=512, pad_token_id=0):
    """
    Pad batch to same length.
    
    Args:
        batch: List of dicts with 'input_ids' and 'labels'
        max_len: Maximum sequence length
        pad_token_id: Padding token ID
    
    Returns:
        dict with padded tensors
    """
    # Find max length in this batch
    max_length = min(max_len, max([len(item['input_ids']) for item in batch]))
    
    batch_size = len(batch)
    padded_input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = min(len(item['input_ids']), max_length)
        padded_input_ids[i, :seq_len] = torch.tensor(item['input_ids'][:seq_len])
        attention_mask[i, :seq_len] = 1
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def train_and_evaluate(model, tokenizer, task, args):
    """Train and evaluate model on a GLUE task."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating on {task.upper()}")
    print(f"{'='*70}\n")
    
    # Load dataset
    if task in ['boolq', 'multirc']:
        dataset = datasets.load_dataset('super_glue', task)
    else:
        # Handle special case for SST-2
        task_name = 'sst2' if task == 'sst2' else task
        dataset = datasets.load_dataset('glue', task_name)
    
    # Limit dataset size for faster eval
    train_size = min(args.train_size, len(dataset['train']))
    val_size = min(args.val_size, len(dataset['validation']))
    
    dataset['train'] = dataset['train'].select(range(train_size))
    dataset['validation'] = dataset['validation'].select(range(val_size))
    
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Tokenize
    tokenize_fn = partial(tokenize_function, tokenizer=tokenizer, task=task)
    dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset['train'].column_names)
    
    # Create dataloaders
    collate_fn = partial(padding_collate_fn, max_len=512, pad_token_id=tokenizer.pad_token_id)
    train_dataloader = torch.utils.data.DataLoader(
        dataset['train'], 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset['validation'], 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    best_metric = 0.0
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.max_epochs}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Evaluate
        model.eval()
        metric = evaluate(model, val_dataloader, task)
        
        metric_name = "F1" if task == 'mrpc' else "Accuracy"
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val {metric_name}: {metric:.4f}")
        
        # Early stopping
        if metric > best_metric:
            best_metric = metric
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest {metric_name}: {best_metric:.4f}")
    return best_metric


def evaluate(model, dataloader, task):
    """Evaluate model on validation set."""
    correct = 0
    total = 0
    tp, fp, fn = 0, 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = outputs['logits'].argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
            
            # For F1 calculation
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
    
    # Return F1 for MRPC, accuracy for others
    if task == 'mrpc':
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return f1
    else:
        accuracy = correct / total if total > 0 else 0.0
        return accuracy


def main():
    parser = argparse.ArgumentParser(description="GLUE evaluation for morphology_clean_tokenizer")
    parser.add_argument('--model_path', type=str, default='outputs/morphology_clean_tokenizer',
                       help='Path to model directory')
    parser.add_argument('--task', type=str, choices=GLUE_TASKS + ['all'], default='all',
                       help='GLUE task to evaluate (or "all")')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=5,
                       help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=2,
                       help='Early stopping patience')
    parser.add_argument('--train_size', type=int, default=5000,
                       help='Max training samples')
    parser.add_argument('--val_size', type=int, default=1000,
                       help='Max validation samples')
    parser.add_argument('--output', type=str, default='outputs/glue_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    print(f"Device: {DEVICE}\n")
    
    # Load tokenizer - check for direct spm.model first, then nested path
    import os
    if os.path.exists(f"{args.model_path}/spm.model"):
        tokenizer_path = f"{args.model_path}/spm.model"
    else:
        tokenizer_path = f"{args.model_path}/morphology_clean_tokenizer/spm.model"
    
    tokenizer = SentencePieceTokenizerWrapper(tokenizer_path)
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}\n")
    
    # Determine tasks to run
    tasks = GLUE_TASKS if args.task == 'all' else [args.task]
    
    results = {}
    
    for task in tasks:
        # Create model for this task
        model = GPT2ForSequenceClassification(
            args.model_path, 
            num_labels=TASK_NUM_LABELS[task]
        ).to(DEVICE)
        
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train and evaluate
        metric = train_and_evaluate(model, tokenizer, task, args)
        results[task] = float(metric)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")
    
    for task, score in results.items():
        metric_name = "F1" if task == 'mrpc' else "Accuracy"
        print(f"{task.upper()}: {score:.4f} ({metric_name})")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
