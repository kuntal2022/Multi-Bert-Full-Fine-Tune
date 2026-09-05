# Fine-Tuning BERT and Its Distilled Variants for Fake News Detection

A comparative fine-tuning project — training and evaluating four BERT-family models (`bert-base-uncased`, `distilbert-base-uncased`, `google/mobilebert-uncased`, and a tiny BERT variant) on a **real vs. fake news classification task**, side-by-side, using a shared training pipeline built with HuggingFace `transformers`.

> **Note on dataset size:** This project uses a small subset (100 rows per split) of the full dataset. The goal here was to build and validate the complete multi-model training pipeline — data loading, tokenization across different tokenizers, class-imbalance handling, custom Trainer logic, early stopping, and evaluation — rather than to produce production-grade accuracy numbers. The same pipeline scales directly to the full dataset by removing the row-limit at the loading step.

## What This Project Does

1. Loads a labeled news dataset (`title`, `text`, `label` columns) with binary real/fake labels.
2. Tokenizes the same data separately for each of the four models (since each has its own tokenizer/vocabulary).
3. Loads each model with `AutoModelForSequenceClassification`, attaching a fresh classification head.
4. Trains all four models using a shared, reusable `CreateTrainer()` function, with:
   - `EarlyStoppingCallback` to prevent overfitting
   - Per-epoch evaluation (accuracy, precision, recall, F1)
   - Automatic best-checkpoint loading
5. Saves each fine-tuned model and its tokenizer separately.
6. Compares performance across all four models (accuracy/F1/training time) to see the practical trade-off between model size and performance.

## Models Compared

| Model | Parameters (approx.) | Purpose in this comparison |
|---|---|---|
| `bert-base-uncased` | 110M | Baseline — full-size reference model |
| `distilbert-base-uncased` | 66M | Knowledge-distilled, ~40% smaller/faster |
| `google/mobilebert-uncased` | 25M | Optimized for mobile/edge deployment |
| Tiny BERT variant | ~4M | Smallest model — speed vs. accuracy trade-off |

## Pipeline Overview

```
Load dataset (100-row subset per split: train / validation / test)
        |
Tokenize per model (each model uses its own tokenizer)
        |
Load each model with AutoModelForSequenceClassification
        |
CreateTrainer() -- shared TrainingArguments + EarlyStoppingCallback
        |
Train all 4 models in a loop
        |
Evaluate -- accuracy, precision, recall, F1, confusion matrix
        |
Save each model + tokenizer to ./saved_models/{model_name}_finetuned
```

## Key Engineering Details

- **Per-model tokenization**: Each model has its own tokenizer and vocabulary, so the dataset is tokenized four separate times rather than once -- reusing one tokenizer's output across different model architectures would silently break training.
- **Fixed-length padding/truncation**: Tokenization uses `padding="max_length"` and `truncation=True` to guarantee consistent tensor shapes across batches (a dynamic/relative padding setup caused a batch-collation error specifically with MobileBERT during development).
- **Custom Trainer for class imbalance** *(where used)*: A `WeightedTrainer` subclass overrides `compute_loss()` to apply `sklearn`'s `compute_class_weight(class_weight='balanced')` inside `CrossEntropyLoss`, so minority classes are penalized more heavily during training.
- **Early stopping**: `EarlyStoppingCallback(early_stopping_patience=2)` paired with `load_best_model_at_end=True` and matching `eval_strategy` / `save_strategy` ("epoch") ensures training stops once validation loss stops improving, and the best checkpoint (not the last one) is what gets saved.

## Results

Each model's classification report (precision/recall/F1 per class) and confusion matrix are generated after training -- see the notebook for the full per-model breakdown and a side-by-side accuracy/training-time comparison.

## Tech Stack

- Python, PyTorch
- HuggingFace `transformers` (`AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, `EarlyStoppingCallback`)
- HuggingFace `datasets`
- scikit-learn (`classification_report`, `confusion_matrix`, `compute_class_weight`)
- seaborn / matplotlib (confusion matrix visualization)

## Setup

```bash
pip install -r requirements.txt
```

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```
*(use `cu128` instead of `cu121` if running on a Blackwell-architecture GPU)*

## Next Steps

- Re-run the full pipeline on the complete dataset (not the 100-row subset) for production-grade metrics.
- Add per-model inference-latency benchmarking to make the size-vs-speed trade-off concrete.
- Push the best-performing fine-tuned model to the HuggingFace Hub.
