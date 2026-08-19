"""
Trains the Fake News Detection model.

Run from the project root:
    python model/train_model.py

What it does:
1. Auto-detects every dataset present in data/ and merges them:
   - ISOT (data/Fake.csv + data/True.csv)
   - WELFake (data/WELFake_Dataset.csv)
   - LIAR (data/liar/train.tsv etc.)
   Falls back to data/sample_data.csv (auto-generated demo data) if none
   of the above are found. See data/README.md for download links.
2. Cleans text (model/preprocess.py — lowercase, strip noise, remove
   stopwords, lemmatize).
3. Vectorizes with TF-IDF (unigrams + bigrams, top 5000 features).
4. Trains and compares 3 classic, CPU-friendly models:
   Logistic Regression, Multinomial Naive Bayes, Linear SVM (Passive
   Aggressive). Picks the best on held-out accuracy.
5. Saves:
   - model/model.pkl          (best trained classifier)
   - model/vectorizer.pkl     (fitted TF-IDF vectorizer)
   - model/metrics.json       (accuracy / precision / recall / f1 / dataset stats)
   - static/img/*.png         (confusion matrix + model comparison charts,
                                used by the Analytics page)
"""
import json
import os
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMG_DIR = os.path.join(BASE_DIR, "static", "img")
os.makedirs(IMG_DIR, exist_ok=True)


def _load_isot():
    """ISOT Fake and Real News Dataset — data/Fake.csv + data/True.csv"""
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        return None

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"
    for d in (fake_df, true_df):
        if "title" in d.columns and "text" in d.columns:
            d["text"] = d["title"].fillna("") + ". " + d["text"].fillna("")
    combined = pd.concat([fake_df[["text", "label"]], true_df[["text", "label"]]],
                          ignore_index=True)
    return combined, "ISOT Fake and Real News Dataset"


def _load_welfake():
    """WELFake Dataset — a single CSV with title/text/label columns.
    Accepts a few common filenames since Kaggle downloads vary."""
    candidates = ["WELFake_Dataset.csv", "WELFake.csv", "welfake.csv", "WELFake_dataset.csv"]
    path = next((os.path.join(DATA_DIR, c) for c in candidates
                 if os.path.exists(os.path.join(DATA_DIR, c))), None)
    if not path:
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "label" not in df.columns or ("text" not in df.columns and "title" not in df.columns):
        print(f"Warning: {os.path.basename(path)} found but columns look unexpected "
              f"({list(df.columns)}) — skipping this file.")
        return None

    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    elif "title" in df.columns:
        df["text"] = df["title"]

    # WELFake's label convention is 1=REAL, 0=FAKE. Guard against the
    # (rare) case a different copy of the CSV flips this by checking
    # the values are the expected {0,1} before mapping.
    unique_labels = set(pd.Series(df["label"]).dropna().unique().tolist())
    if unique_labels <= {0, 1, 0.0, 1.0}:
        df["label"] = df["label"].map({1: "REAL", 0: "FAKE", 1.0: "REAL", 0.0: "FAKE"})
    else:
        print(f"Warning: unexpected label values in WELFake file ({unique_labels}) — skipping.")
        return None

    return df[["text", "label"]], "WELFake Dataset"


def _load_liar():
    """LIAR Dataset — Politifact statements with 6-way truthfulness labels,
    collapsed to binary REAL/FAKE. Expects data/liar/train.tsv (and
    optionally test.tsv, valid.tsv) in the original LIAR column format."""
    liar_dir = os.path.join(DATA_DIR, "liar")
    if not os.path.isdir(liar_dir):
        return None

    cols = ["id", "label", "statement", "subject", "speaker", "job", "state",
            "party", "barely_true_c", "false_c", "half_true_c", "mostly_true_c",
            "pants_fire_c", "context"]
    frames = []
    for fname in ["train.tsv", "test.tsv", "valid.tsv"]:
        fpath = os.path.join(liar_dir, fname)
        if os.path.exists(fpath):
            try:
                frames.append(pd.read_csv(fpath, sep="\t", header=None, names=cols))
            except Exception as e:
                print(f"Warning: couldn't read {fname} ({e}) — skipping.")
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    label_map = {
        "true": "REAL", "mostly-true": "REAL", "half-true": "REAL",
        "barely-true": "FAKE", "false": "FAKE", "pants-fire": "FAKE",
    }
    df["label"] = df["label"].map(label_map)
    df["text"] = df["statement"]
    df = df.dropna(subset=["text", "label"])
    return df[["text", "label"]], "LIAR Dataset (Politifact statements)"


def load_data():
    """Auto-detects every supported dataset present in data/ and merges
    them into one combined training set. Add more files any time —
    no code changes needed, just drop them in data/ and re-run training.
    Falls back to the synthetic demo dataset if nothing real is found."""
    loaders = [_load_isot, _load_welfake, _load_liar]
    loaded = []

    for loader in loaders:
        result = loader()
        if result is not None:
            df, name = result
            df = df.dropna(subset=["text", "label"])
            df = df[df["text"].astype(str).str.strip() != ""]
            if len(df):
                loaded.append((df, name))
                print(f"Loaded {name}: {len(df)} articles")

    if not loaded:
        sample_path = os.path.join(DATA_DIR, "sample_data.csv")
        if not os.path.exists(sample_path):
            print("No dataset found. Generating the demo dataset now...")
            sys.path.append(DATA_DIR)
            import generate_sample_data
            generate_sample_data.main()
        print("No real dataset found — using data/sample_data.csv (demo "
              "dataset). See data/README.md to use real datasets instead.")
        df = pd.read_csv(sample_path)
        source = "Synthetic demo dataset (data/generate_sample_data.py)"
    else:
        df = pd.concat([d for d, _ in loaded], ignore_index=True)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["text"])
        if before_dedup != len(df):
            print(f"Removed {before_dedup - len(df)} duplicate articles across sources.")
        source = " + ".join(name for _, name in loaded)
        if len(loaded) > 1:
            print(f"Combined {len(loaded)} datasets into one training set.")

    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df, source


def main():
    start = time.time()
    df, source = load_data()
    print(f"Dataset loaded: {len(df)} articles "
          f"({(df['label'] == 'REAL').sum()} real / {(df['label'] == 'FAKE').sum()} fake)")

    print("Cleaning text (lowercase, remove stopwords, lemmatize)...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    avg_len_before = df["text"].str.split().apply(len).mean()
    avg_len_after = df["clean_text"].str.split().apply(len).mean()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42,
        stratify=df["label"]
    )

    print("Vectorizing with TF-IDF (max 5000 features, unigrams+bigrams)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Passive Aggressive (Linear SVM-like)": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    }

    results = {}
    trained_models = {}
    print("\nTraining & evaluating models on CPU...")
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="FAKE")
        rec = recall_score(y_test, preds, pos_label="FAKE")
        f1 = f1_score(y_test, preds, pos_label="FAKE")
        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "train_time_sec": round(time.time() - t0, 2),
        }
        trained_models[name] = clf
        print(f"  {name:38s} acc={acc*100:5.2f}%  f1={f1*100:5.2f}%  "
              f"({time.time()-t0:.1f}s)")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} ({results[best_name]['accuracy']}% accuracy)")

    # Logistic Regression is preferred as the deployed model when close in
    # accuracy, since it gives well-calibrated predict_proba() confidence
    # scores (needed for the Detect page). Only override if another model
    # clearly outperforms it.
    lr_acc = results["Logistic Regression"]["accuracy"]
    best_acc = results[best_name]["accuracy"]
    if best_name != "Logistic Regression" and (best_acc - lr_acc) < 1.0:
        best_name = "Logistic Regression"
        best_model = trained_models["Logistic Regression"]
        print(f"(Deploying Logistic Regression instead — within 1% of best "
              f"and supports confidence scores.)")

    # --- Save model + vectorizer ---
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    # --- Charts for the Analytics page ---
    preds_best = best_model.predict(X_test_vec)
    cm = confusion_matrix(y_test, preds_best, labels=["REAL", "FAKE"])

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"],
                cbar=False, annot_kws={"size": 14})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=140)
    plt.close()

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(range(len(names)), accs, color=["#5B8DEF", "#3DDC97", "#FFB454"])
    plt.xticks(range(len(names)), [n.replace(" (Linear SVM-like)", "") for n in names],
                rotation=15, ha="right", fontsize=9)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Model Comparison")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc}%",
                  ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=140)
    plt.close()

    label_counts = df["label"].value_counts().to_dict()
    plt.figure(figsize=(5, 4.5))
    plt.bar(label_counts.keys(), label_counts.values(),
            color=["#3DDC97" if k == "REAL" else "#FF5D5D" for k in label_counts])
    plt.ylabel("Number of Articles")
    plt.title("Dataset Class Balance")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "class_balance.png"), dpi=140)
    plt.close()

    # --- Confidence calibration (reliability diagram) ---
    # Shows whether the model's confidence scores can be trusted: e.g. when
    # it says "90% confident", is it actually right about 90% of the time?
    # This is a stronger honesty check than accuracy alone.
    if hasattr(best_model, "predict_proba"):
        print("Generating calibration (reliability) diagram...")
        probs = best_model.predict_proba(X_test_vec)
        classes = list(best_model.classes_)
        fake_idx = classes.index("FAKE")
        y_true_binary = (y_test == "FAKE").astype(int)
        prob_fake = probs[:, fake_idx]

        frac_pos, mean_pred = calibration_curve(y_true_binary, prob_fake, n_bins=10)

        plt.figure(figsize=(5.5, 4.5))
        plt.plot([0, 1], [0, 1], linestyle="--", color="#8592AB", label="Perfectly calibrated")
        plt.plot(mean_pred, frac_pos, marker="o", color="#5B8DEF", label=best_name)
        plt.xlabel("Predicted probability of FAKE")
        plt.ylabel("Actual fraction that were FAKE")
        plt.title("Confidence Calibration")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "calibration.png"), dpi=140)
        plt.close()
        has_calibration = True
    else:
        has_calibration = False

    # --- Word clouds (real vs fake) for the Analytics page ---
    print("Generating word clouds...")
    for label, fname, cmap in [("REAL", "wordcloud_real.png", "Greens"),
                                 ("FAKE", "wordcloud_fake.png", "Reds")]:
        text_blob = " ".join(df[df["label"] == label]["clean_text"].tolist())
        if text_blob.strip():
            wc = WordCloud(width=900, height=500, background_color="#0B1220",
                            colormap=cmap, max_words=120).generate(text_blob)
            wc.to_file(os.path.join(IMG_DIR, fname))

    # --- Save metrics.json (read by the Flask app for the Analytics page) ---
    metrics = {
        "dataset_source": source,
        "total_articles": int(len(df)),
        "real_count": int(label_counts.get("REAL", 0)),
        "fake_count": int(label_counts.get("FAKE", 0)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "avg_words_before_cleaning": round(float(avg_len_before), 1),
        "avg_words_after_cleaning": round(float(avg_len_after), 1),
        "vocabulary_size": len(vectorizer.get_feature_names_out()),
        "deployed_model": best_name,
        "all_models": results,
        "deployed_metrics": results[best_name],
        "confusion_matrix": {
            "labels": ["REAL", "FAKE"],
            "matrix": cm.tolist(),
        },
        "training_time_sec": round(time.time() - start, 2),
        "has_calibration_chart": has_calibration,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model/model.pkl, model/vectorizer.pkl, model/metrics.json")
    print(f"Saved charts to static/img/")
    print(f"Total time: {time.time() - start:.1f}s")
    print("\nDone! Now run: python app.py")


if __name__ == "__main__":
    main()
