# evaluate_glue.py
import argparse
import datasets
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer#, AutoConfig
from functools import partial
from tqdm import tqdm


# GLUE_SUBSETS = ['cola','sst2', 'mrpc', 'qnli', 'rte', 'boolq', 'multirc']
GLUE_SUBSETS = ['boolq', 'mrpc', 'multirc', 'qqp']#, 'mnli', 'rte', 'wsc']
INPUTS = {
    # 'cola': ['sentence'],
    # 'sst2': ['sentence'],
    'boolq': ['question', 'passage'],
    # 'mnli': ['premise', 'hypothesis'],
    'mrpc': ['sentence1', 'sentence2'],
    # 'qnli': ['question', 'sentence'],
    # 'rte': ['sentence1', 'sentence2'],
    'multirc': ['paragraph', 'question', 'answer'],
    'qqp': ['question1', 'question2'],
}
DEVICE = 'cuda'


parser = argparse.ArgumentParser()
parser.add_argument('--subset', type=str, required=True, choices=GLUE_SUBSETS)
parser.add_argument('--model_type', type=str, required=True, choices=['encoder', 'decoder'])
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--bs', '--batch_size', type=int, default=64)
parser.add_argument('--lr', '--learning_rate', type=float, default=5e-5)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)

def tokenize(examples, tokenizer, subset, truncate=True):
    batch = {
        "input_ids": [],
        "labels": [],
    }

    for i in range(len(examples['label'])):
        input_txt = " ".join([examples[txt][i] for txt in INPUTS[subset]])
        input_ids = tokenizer.encode(input_txt, truncation=truncate)
        batch["input_ids"].append(input_ids)
        batch["labels"].append([examples['label'][i]])

    return batch

def padding_collate_fn(batch, max_len=1024, left_padding=False):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            if left_padding:
                padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])
            else:
                padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch


def main():
    args = parser.parse_args()
    model_name = "prajjwal1/bert-tiny" if args.model_type == "encoder" else "sshleifer/tiny-gpt2"
    if not args.model_path:
        args.model_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.model_max_length = 512 # will truncate the inputs to this length
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, subset=args.subset) # need to make the function unary for map()

    if args.subset in ['boolq', 'multirc']:
        dataset = datasets.load_dataset('super_glue', args.subset)
    else:
        dataset = datasets.load_dataset('glue', args.subset)

    # filter down to 5k samples for training and 2k samples for validation
    dataset['train'] = dataset['train'].select(range(10000))
    dataset['validation'] = dataset['validation'].select(range(2000))
    if 'test' in dataset:
        del dataset['test']

    dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset['train'].column_names) 

    # config = AutoConfig.from_pretrained(args.model_path, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2).to(DEVICE)
    model.config.pad_token_id = 0 # NOTE: can remove, only needed for testing tiny-gpt2

    train(model, dataset, args)


def train(model, dataset, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.bs, shuffle=True, collate_fn=padding_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(dataset['validation'], batch_size=args.bs, shuffle=False, collate_fn=padding_collate_fn)

    best = 0.0
    patience = args.patience
    for epoch in range(args.max_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            inputs = batch['input_ids'].to(device=DEVICE)
            labels = batch['labels'].to(device=DEVICE)
            outputs = model(inputs, labels=labels, attention_mask=inputs != 0)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        result = evaluate(model, valid_dataloader, args.subset)
        print(f"Epoch: {epoch}, Result: {result}")

        if result > best:
            best = result
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                break

    print(f"Best result: {best}")


def evaluate(model, dataloader, subset):
    correct = 0.0
    total = 0.0

    tp, fp, fn = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device=DEVICE)
            labels = batch['labels'][:, 0].to(device=DEVICE)

            outputs = model(input_ids, attention_mask=input_ids != 0).logits.argmax(-1)

            tp += ((outputs == 1) & (labels == 1)).sum().item()
            fp += ((outputs == 1) & (labels == 0)).sum().item()
            fn += ((outputs == 0) & (labels == 1)).sum().item()
            correct += (outputs == labels).sum().item()
            total += len(labels)

    f1 = 2 * tp / (2 * tp + fp + fn)
    if subset == 'mrpc':
        return f1
    else:
        return correct / total



if __name__ == '__main__':
    main()