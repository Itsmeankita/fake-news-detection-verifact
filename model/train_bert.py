"""
OPTIONAL — trains a DistilBERT classifier on the same dataset, purely for
comparison against the deployed TF-IDF + Logistic Regression model.

Why this is separate from train_model.py:
- The main pipeline is deliberately TF-IDF + classic ML: fast, CPU-only,
  and easy to explain — matching the "no GPU needed" project brief.
- This script is for anyone who wants to *additionally* show deep learning
  skills (e.g. for a resume line, or comparing approaches in a report).
- It needs extra heavy dependencies (torch, transformers) and real compute
  time — several hours on CPU, or ~15-20 minutes on a free Colab/Kaggle GPU.
  It is NOT required to run the website; the app only uses model.pkl.

Setup (only if you want to run this):
    pip install torch transformers

Usage:
    python model/train_bert.py

What it does:
- Loads the same dataset as train_model.py (Fake.csv/True.csv, or the
  sample fallback)
- Fine-tunes 'distilbert-base-uncased' for binary classification
- Evaluates on the same held-out test split logic
- Prints accuracy/precision/recall/F1 and appends the result into
  model/metrics.json under "all_models" as "DistilBERT (deep learning)",
  so it shows up automatically in the Analytics page's comparison table
- Saves the fine-tuned model to model/bert_model/ (not loaded by the
  website — this is a comparison artifact, not the deployed model)
"""
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (DistilBertForSequenceClassification,
                               DistilBertTokenizerFast, Trainer,
                               TrainingArguments)
except ImportError:
    print("This script needs extra packages not in requirements.txt "
          "(they're heavy and optional). Install them first with:\n\n"
          "    pip install torch transformers\n")
    sys.exit(1)

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.append(MODEL_DIR)
from train_model import load_data  # reuse the same data-loading logic


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    start = time.time()
    df, source = load_data()

    # A random subsample keeps CPU training time reasonable for a demo/report;
    # remove the .sample() line to train on the full dataset if you have a GPU.
    if len(df) > 6000:
        df = df.sample(n=6000, random_state=42).reset_index(drop=True)
        print(f"Using a 6,000-row subsample of {source} for feasible CPU/GPU training time.")

    label_map = {"REAL": 0, "FAKE": 1}
    df["label_id"] = df["label"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label_id"].tolist(), test_size=0.2,
        random_state=42, stratify=df["label_id"]
    )

    print("Loading DistilBERT tokenizer + model (downloads ~260MB on first run)...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)

    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=256)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=256)

    train_dataset = NewsDataset(train_enc, y_train)
    test_dataset = NewsDataset(test_enc, y_test)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "bert_checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning DistilBERT (this is the slow part)...")
    trainer.train()
    results = trainer.evaluate()
    print("Results:", results)

    save_path = os.path.join(MODEL_DIR, "bert_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved fine-tuned model to {save_path}")

    # Append into metrics.json so it shows on the Analytics comparison table
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics.setdefault("all_models", {})["DistilBERT (deep learning)"] = {
            "accuracy": round(results["eval_accuracy"] * 100, 2),
            "precision": round(results["eval_precision"] * 100, 2),
            "recall": round(results["eval_recall"] * 100, 2),
            "f1_score": round(results["eval_f1"] * 100, 2),
            "train_time_sec": round(time.time() - start, 2),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Added DistilBERT results to model/metrics.json — "
              "restart the app to see it on the Analytics page.")


if __name__ == "__main__":
    main()
