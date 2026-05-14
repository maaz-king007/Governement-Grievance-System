"""
train_models.py
Train all three classification models:
  1. Department classifier  (7 classes)
  2. Sentiment classifier   (negative / neutral / positive)
  3. Urgency classifier     (high / medium / low)

Pipeline: TF-IDF (char + word n-grams) → LinearSVC / LogisticRegression
This closely mirrors the feature engineering used with IndicBERT fine-tuning
but runs fully offline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from utils.preprocessor import preprocess

DATA_PATH = "data/grievances.csv"
MODELS_DIR = "models"


def build_tfidf_pipeline(estimator):
    """TF-IDF with both word and character n-grams for Hinglish."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=20000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", estimator),
    ])


def train_department_classifier(df):
    print("\n🏢 Training Department Classifier...")
    X = df["text_clean"].values
    y = df["department"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = build_tfidf_pipeline(LinearSVC(C=1.0, max_iter=2000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le, acc


def train_sentiment_classifier(df):
    print("\n💬 Training Sentiment Classifier...")
    X = df["text_clean"].values
    y = df["sentiment"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = build_tfidf_pipeline(
        LogisticRegression(C=2.0, max_iter=1000)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le, acc


def train_urgency_classifier(df):
    print("\n🚨 Training Urgency Classifier...")
    X = df["text_clean"].values
    y = df["urgency"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = build_tfidf_pipeline(LinearSVC(C=0.5, max_iter=2000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le, acc


def train_abusive_classifier(df):
    print("\n🚫 Training Abusive Language Detector...")
    X = df["text_clean"].values
    y = df["is_abusive"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_tfidf_pipeline(LogisticRegression(C=1.0, max_iter=500))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")

    return model, acc


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("📂 Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print("   Dataset not found, generating...")
        import subprocess
        subprocess.run(["python3", "data/generate_dataset.py"])

    df = pd.read_csv(DATA_PATH)
    print(f"   {len(df)} records loaded")
    print(f"   Departments: {df['department'].nunique()}")
    print(f"   Class distribution:\n{df['department'].value_counts()}")

    # ── Preprocess ─────────────────────────────────────────────────────────────
    print("\n🔄 Preprocessing text...")
    df["text_clean"] = df["text"].apply(preprocess)

    # ── Train models ───────────────────────────────────────────────────────────
    dept_model, dept_le, dept_acc = train_department_classifier(df)
    sent_model, sent_le, sent_acc = train_sentiment_classifier(df)
    urg_model, urg_le, urg_acc = train_urgency_classifier(df)
    abuse_model, abuse_acc = train_abusive_classifier(df)

    # ── Save models ────────────────────────────────────────────────────────────
    print("\n💾 Saving models...")
    with open(f"{MODELS_DIR}/dept_model.pkl", "wb") as f:
        pickle.dump(dept_model, f)
    with open(f"{MODELS_DIR}/dept_le.pkl", "wb") as f:
        pickle.dump(dept_le, f)

    with open(f"{MODELS_DIR}/sent_model.pkl", "wb") as f:
        pickle.dump(sent_model, f)
    with open(f"{MODELS_DIR}/sent_le.pkl", "wb") as f:
        pickle.dump(sent_le, f)

    with open(f"{MODELS_DIR}/urg_model.pkl", "wb") as f:
        pickle.dump(urg_model, f)
    with open(f"{MODELS_DIR}/urg_le.pkl", "wb") as f:
        pickle.dump(urg_le, f)

    with open(f"{MODELS_DIR}/abuse_model.pkl", "wb") as f:
        pickle.dump(abuse_model, f)

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "departments": list(dept_le.classes_),
        "sentiments": list(sent_le.classes_),
        "urgencies": list(urg_le.classes_),
        "accuracies": {
            "department": round(dept_acc, 4),
            "sentiment": round(sent_acc, 4),
            "urgency": round(urg_acc, 4),
            "abusive": round(abuse_acc, 4),
        }
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ All models saved!")
    print(f"   Department accuracy : {dept_acc:.2%}")
    print(f"   Sentiment accuracy  : {sent_acc:.2%}")
    print(f"   Urgency accuracy    : {urg_acc:.2%}")
    print(f"   Abusive accuracy    : {abuse_acc:.2%}")


if __name__ == "__main__":
    main()
