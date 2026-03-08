# Morphology-Aware Tokenization and Pairwise Contrastive Fine-Tuning for Low-Resource Language Modeling

**Authors:** Felix Aker ([felix.aker@tum.de](mailto:felix.aker@tum.de)), Begüm Kara ([begum.kara@tum.de](mailto:begum.kara@tum.de))  
**Project:** BabyLM Challenge (Team 2)

---

## 🚀 Overview
This repository contains the methodology and codebase for training data-efficient language models on the BabyLM challenge datasets (1M and 10M tokens). Our approach focuses on three core pillars: **Rigorous Data Curation**, **Morphology-Aware Tokenization**, and **Pairwise Contrastive Fine-Tuning**. 

We achieve a **BLiMP score of 62.14%** on the 1M dataset and a **70.79% average on GLUE**, demonstrating that intentional structural biases can significantly improve syntactic competence in extremely low-resource regimes.

---

## 🛠 Methodology

### 1. Data Curation (4-Phase Pipeline)
We prioritize high-quality text over sheer quantity. Our pipeline filters raw corpora down to a clean subset (~37.5% retention for 1M):
- **Phase 1: Per-Corpus Cleaning**: Removal of timestamp markers, metadata, and speaker tags.
- **Phase 2: Global Deduplication**: LSH MinHash for near-duplicates and SHA256 for exact matches.
- **Phase 3: Quality Scoring**: A multi-factor scoring function (length, alpha ratio, digit/punctuation balance, repetition).
- **Phase 4: Stratified Sampling**: Balanced mixture (40% Gutenberg, 25% Wiki, etc.) via reservoir sampling.

🔗 **Script:** [scripts/clean_corpus.py](file:///Users/begum/NLP-BabyLM-Team-2/scripts/clean_corpus.py)

### 2. Morphology-Aware Tokenization
Standard BPE often breaks grammatical markers. Our custom tokenizer pre-splits words at morphological boundaries (e.g., `played` → `play @@ed`), preserving suffixes like `-ed`, `-ing`, and `-s`.
- **Finding:** Morphology-aware training is most beneficial in the 1M token setting, providing a strong inductive bias that models scale to implicit learning at 10M tokens.

🔗 **Script:** [scripts/train_morphology_tokenizer.py](file:///Users/begum/NLP-BabyLM-Team-2/scripts/train_morphology_tokenizer.py)

### 3. Pairwise Contrastive Fine-Tuning
To improve syntactic sensitivity, we fine-tune the model to distinguish between grammatically correct/incorrect minimal pairs using **Margin Ranking Loss**.
- **Dataset:** 90/10 mix of natural text and targeted synthetic minimal pairs (Subject-Verb Agreement, Inflection, etc.).
- **Optimizer:** GPT-2 Decoder with joint LM and Ranking objectives.

🔗 **Script:** [scripts/train_pairwise_contrastive.py](file:///Users/begum/NLP-BabyLM-Team-2/scripts/train_pairwise_contrastive.py)

---

## 📊 Metric Stability Research
A key contribution of this project is the analysis of **Tokenizer Granularity Bias**. We demonstrate that standard per-token normalized metrics (Mean Accuracy) are biased toward the training tokenizer.
- **The Solution:** We identify **Bits-per-Byte (BPB)** and **Bits-per-Character (BPC)** as the stable, invariant metrics for cross-model comparisons.
- **Result:** Using BPB and BPC, the Morphology-aware model maintains a decisive lead (**60.19% vs 54.66%**) over standard BPE.

🔗 **Module:** [Evaluation and Metric Stability/](file:///Users/begum/NLP-BabyLM-Team-2/Evaluation%20and%20Metric%20Stability/)

---

## 📁 Repository Structure
```bash
.
├── Data/                       # Cleaned 1M/10M datasets and statistics
├── Tokenizer/                  # Trained morphology-aware BPE models
├── scripts/                    # Core training & cleaning pipeline
│   ├── clean_corpus.py         # 4-Phase data curation
│   ├── train_morphology_tokenizer.py
│   └── train_pairwise_contrastive.py
├── evaluation/                 # ML evaluation scripts (BLiMP, GLUE)
└── Evaluation and Metric Stability/ # Research on tokenizer bias
```

---

## 📈 Key Results (1M Tokens)
| Metric | Standard BPE | Morphology (Ours) | Contrastive Fine-Tuned |
| :--- | :---: | :---: | :---: |
| **BLiMP Accuracy** | 60.54% | 61.81% | **62.14%** |
| **GLUE Average** | - | - | **70.79%** |

---

## 💻 Usage

### Installation
```bash
pip install -r "Evaluation and Metric Stability/requirements.txt"
```

### Reproducing Results
1. **Clean Data:**
   ```bash
   python scripts/clean_corpus.py --input-dir Data/raw --output-dir Data/cleaned
   ```
2. **Train Tokenizer:**
   ```bash
   python scripts/train_morphology_tokenizer.py --train_file Data/cleaned/clean_1M.txt
   ```
3. **Fine-Tune Model:**
   ```bash
   python scripts/train_pairwise_contrastive.py --model_path path/to/gpt2 --pairs_file Data/minimal_pairs.jsonl
   ```

### Finished Models

**You can download the finished models from here:** https://drive.google.com/drive/folders/1SvtCaTcH7EIKQ2nQtmvD-HC9CogcPB9t?usp=sharing