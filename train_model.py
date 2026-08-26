"""
train_model.py
---------------
End-to-end training pipeline for TruthGuard.

Steps:
    1. Load dataset/Fake.csv and dataset/True.csv
    2. Label Fake = 0, Real = 1 and combine into a single dataframe
    3. Clean/preprocess the text (see utils/text_preprocessing.py)
    4. Vectorize with TF-IDF
    5. Train 4 candidate models and evaluate them
    6. Pick the best model (by F1 score) and save it + the vectorizer

Run:
    python train_model.py

Outputs:
    model.pkl            - the best performing trained classifier
    vectorizer.pkl        - the fitted TF-IDF vectorizer
    model_metrics.json    - accuracy/precision/recall/F1/confusion matrix
                            for every model that was trained, for the report
                            shown on the /about page.
"""

import os
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import Config
from utils.text_preprocessing import clean_text


def load_dataset() -> pd.DataFrame:
    """Load Fake.csv / True.csv, label them, and merge into one dataframe."""
    if not (os.path.exists(Config.FAKE_CSV) and os.path.exists(Config.TRUE_CSV)):
        raise FileNotFoundError(
            "dataset/Fake.csv and dataset/True.csv not found.\n"
            "Either download the Kaggle 'Fake and Real News Dataset' into the "
            "dataset/ folder, or run:\n"
            "    python dataset/generate_sample_dataset.py\n"
            "to create a small synthetic dataset for testing the pipeline."
        )

    fake_df = pd.read_csv(Config.FAKE_CSV)
    true_df = pd.read_csv(Config.TRUE_CSV)

    fake_df["label"] = 0  # Fake
    true_df["label"] = 1  # Real

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text (if both columns exist) into one field to analyze
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset must contain a 'text' column (and ideally 'title').")

    df = df[["content", "label"]].dropna()
    df = df[df["content"].str.strip().str.len() > 0]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def evaluate_model(name, model, X_test, y_test):
    """Compute and return a dict of standard classification metrics."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred) * 100, 2),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }


def main():
    print("=" * 60)
    print("TruthGuard - Model Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")
    df = load_dataset()
    print(f"  Loaded {len(df)} articles "
          f"({(df['label'] == 0).sum()} fake / {(df['label'] == 1).sum()} real)")

    print("\n[2/5] Cleaning & preprocessing text (this can take a while)...")
    start = time.time()
    df["clean_content"] = df["content"].apply(clean_text)
    print(f"  Done in {time.time() - start:.1f}s")

    print("\n[3/5] Splitting train/test and vectorizing with TF-IDF...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\n[4/5] Training candidate models...")
    candidates = {
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=50, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42),
    }

    results = []
    trained_models = {}
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train_tfidf, y_train)
        metrics = evaluate_model(name, model, X_test_tfidf, y_test)
        metrics["train_time_seconds"] = round(time.time() - t0, 2)
        results.append(metrics)
        trained_models[name] = model
        print(f"  {name:<32} | acc={metrics['accuracy']:>6}% "
              f"| prec={metrics['precision']:>6}% | rec={metrics['recall']:>6}% "
              f"| f1={metrics['f1_score']:>6}%")

    print("\n[5/5] Selecting best model by F1 score...")
    best = max(results, key=lambda r: r["f1_score"])
    best_model_name = best["model"]
    best_model = trained_models[best_model_name]
    print(f"  Best model: {best_model_name} (F1 = {best['f1_score']}%)")

    joblib.dump(best_model, Config.MODEL_PATH)
    joblib.dump(vectorizer, Config.VECTORIZER_PATH)

    with open(Config.METRICS_PATH, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "trained_on_rows": len(df),
            "results": results,
        }, f, indent=2)

    print(f"\nSaved model to        {Config.MODEL_PATH}")
    print(f"Saved vectorizer to   {Config.VECTORIZER_PATH}")
    print(f"Saved metrics to      {Config.METRICS_PATH}")
    print("\nDone! You can now run the Flask app with: python app.py")


if __name__ == "__main__":
    main()
