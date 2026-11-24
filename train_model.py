"""
Script to train the resume categorization model and save model artifacts.

Usage:
    python train_model.py [--data-path DATA_PATH] [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Add current root directory to PYTHONPATH for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.text_cleaner import clean_resume


def parse_args():
    parser = argparse.ArgumentParser(description="Train Resume Categorization Model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="UpdatedResumeDataSet.csv",
        help="Path to training CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory where trained model artifacts will be saved",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Ratio of data reserved for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed state for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 Resume Screening Model Training Pipeline")
    print("=" * 60)

    if not os.path.exists(args.data_path):
        print(f"❌ Error: Dataset file '{args.data_path}' not found.")
        sys.exit(1)

    print(f"📊 Loading dataset from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"   Loaded {len(df)} total raw samples across {df['Category'].nunique()} categories.")

    print("⚖️ Balancing classes via oversampling...")
    max_size = df["Category"].value_counts().max()
    balanced_dfs = [
        group.sample(max_size, replace=True, random_state=args.random_state)
        for _, group in df.groupby("Category")
    ]
    df = pd.concat(balanced_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    print(f"   Balanced dataset size: {len(df)} total samples.")

    print("🧹 Cleaning resume text...")
    df["Cleaned_Resume"] = df["Resume"].apply(clean_resume)

    print("🏷️ Encoding target categories...")
    le = LabelEncoder()
    df["Category_Encoded"] = le.fit_transform(df["Category"])

    print("🔤 Extracting features using TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(df["Cleaned_Resume"]).toarray()
    y = df["Category_Encoded"].values

    print(f"🔀 Splitting data (train: {1 - args.test_size:.0%}, test: {args.test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print("🧠 Training OneVsRest Support Vector Classifier (SVC)...")
    svc_model = OneVsRestClassifier(SVC(kernel="linear"))
    svc_model.fit(X_train, y_train)

    print("📈 Evaluating model performance...")
    y_pred = svc_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    print(f"✅ Training Accuracy: {svc_model.score(X_train, y_train):.4f}")
    print(f"🎯 Test Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save pickle artifacts
    model_file = os.path.join(args.output_dir, "clf.pkl")
    tfidf_file = os.path.join(args.output_dir, "tfidf.pkl")
    encoder_file = os.path.join(args.output_dir, "encoder.pkl")
    metrics_file = os.path.join(args.output_dir, "metrics.json")

    with open(model_file, "wb") as f:
        pickle.dump(svc_model, f)
    with open(tfidf_file, "wb") as f:
        pickle.dump(tfidf, f)
    with open(encoder_file, "wb") as f:
        pickle.dump(le, f)

    # Also save copies to root for backward compatibility
    with open("clf.pkl", "wb") as f:
        pickle.dump(svc_model, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "test_accuracy": float(accuracy),
        "total_samples": len(df),
        "num_classes": len(le.classes_),
        "classes": list(le.classes_),
        "classification_report": report_dict,
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("-" * 60)
    print(f"🎉 Model artifacts successfully saved to '{args.output_dir}/' and root:")
    print(f"   ├── {model_file}")
    print(f"   ├── {tfidf_file}")
    print(f"   ├── {encoder_file}")
    print(f"   └── {metrics_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
