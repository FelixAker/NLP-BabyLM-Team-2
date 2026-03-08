from transformers import set_seed, AutoConfig, AutoModelForCausalLM, DebertaV2Tokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import os
import argparse

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.flatten()
    labels = labels.flatten()
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]
    correct = labels == predictions
    accuracy = correct.sum() / float(len(correct)) if len(correct) > 0 else 0.0
    return {"acc": accuracy}

def train_model(vocab_size, tokenizer_path, output_dir, epochs=10, smoke_test=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load custom tokenizer
    tokenizer = DebertaV2Tokenizer(tokenizer_path)
    
    # Create configuration
    model_name = "openai-community/gpt2"
    config = AutoConfig.from_pretrained(model_name)
    config.hidden_size = 384
    config.intermediate_size = 1280
    config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    model = AutoModelForCausalLM.from_config(config)
    
    # Load datasets
    dataset = load_dataset('text', data_files={'train': 'data/train.txt', 'validation': 'data/dev.txt'})
    
    if smoke_test:
        dataset['train'] = dataset['train'].select(range(100))
        epochs = 1

    set_seed(0)

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset = dataset['validation'],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        args = SFTConfig(
            output_dir = output_dir,
            remove_unused_columns = True,
            label_names = ["labels"],
            dataset_num_proc = 1 if smoke_test else 12,
            packing = not smoke_test,
            eval_packing = not smoke_test,
            max_length = 64,
            dataset_text_field = "text",
            eval_strategy = "steps",
            per_device_train_batch_size = 64,
            gradient_accumulation_steps = 1,
            warmup_ratio = 0.05,
            num_train_epochs = epochs,
            learning_rate = 2e-4,
            fp16 = True,
            optim = "adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            max_grad_norm=1,
            logging_steps = 10,
            eval_steps = 100,
            save_steps = 1000,
            report_to = "none",
            seed = 0,
        ),
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Finished training model with vocab_size={vocab_size} in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()
    
    train_model(args.vocab_size, args.tokenizer_path, args.output_dir, args.epochs, args.smoke_test)
