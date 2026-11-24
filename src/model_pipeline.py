"""
Machine learning prediction pipeline and artifact loader for Resume Classification.
"""

import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.text_cleaner import clean_resume


class ResumeClassifier:
    """
    Encapsulates the pre-trained TF-IDF vectorizer, Label Encoder,
    and OneVsRest Support Vector Machine (SVC) model for resume classification.
    """

    def __init__(self, model_dir: str = "."):
        """
        Initializes the ResumeClassifier by locating and loading pickle artifacts.

        Args:
            model_dir (str): Directory where clf.pkl, tfidf.pkl, encoder.pkl are saved.
        """
        self.model_dir = model_dir
        self.svc_model = None
        self.tfidf = None
        self.encoder = None
        self.is_loaded = False
        self._load_artifacts()

    def _find_file(self, filename: str) -> str:
        """Helper to search for model artifact in model_dir, 'models/' subfolder, or root."""
        candidates = [
            os.path.join(self.model_dir, filename),
            os.path.join(self.model_dir, "models", filename),
            os.path.join(".", "models", filename),
            os.path.join(".", filename),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"Artifact '{filename}' not found in paths: {candidates}")

    def _load_artifacts(self) -> None:
        """Loads model, vectorizer, and label encoder from disk."""
        try:
            model_path = self._find_file("clf.pkl")
            tfidf_path = self._find_file("tfidf.pkl")
            encoder_path = self._find_file("encoder.pkl")

            with open(model_path, "rb") as f:
                self.svc_model = pickle.load(f)
            with open(tfidf_path, "rb") as f:
                self.tfidf = pickle.load(f)
            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)

            self.is_loaded = True
        except FileNotFoundError as e:
            self.is_loaded = False
            # Defer error handling to caller (e.g. Streamlit app will prompt user to train model first)

    def predict(self, raw_resume_text: str) -> str:
        """
        Predicts the single top category for a given raw resume text.

        Args:
            raw_resume_text (str): Extracted resume content.

        Returns:
            str: Predicted job category name.
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts are not loaded. Please train the model using 'python train_model.py'.")

        cleaned = clean_resume(raw_resume_text)
        vectorized = self.tfidf.transform([cleaned]).toarray()
        pred_label = self.svc_model.predict(vectorized)
        category_name = self.encoder.inverse_transform(pred_label)[0]
        return category_name

    def predict_top_k(self, raw_resume_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predicts top K categories with confidence percentages.

        Args:
            raw_resume_text (str): Extracted resume content.
            top_k (int): Number of top matches to return.

        Returns:
            List[Tuple[str, float]]: List of (category_name, confidence_percentage) sorted by confidence.
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts are not loaded.")

        cleaned = clean_resume(raw_resume_text)
        vectorized = self.tfidf.transform([cleaned]).toarray()

        # Extract decision function scores from OneVsRestClassifier
        if hasattr(self.svc_model, "decision_function"):
            decision_scores = self.svc_model.decision_function(vectorized)[0]
            # Softmax transformation for normalized probability scores
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / np.sum(exp_scores)
        else:
            # Fallback to simple prediction
            pred_class = self.svc_model.predict(vectorized)[0]
            probabilities = np.zeros(len(self.encoder.classes_))
            probabilities[pred_class] = 1.0

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            cat_name = self.encoder.classes_[idx]
            conf = float(probabilities[idx] * 100.0)
            results.append((cat_name, round(conf, 2)))

        return results

    def get_all_categories(self) -> List[str]:
        """Returns the list of all supported category names."""
        if not self.is_loaded:
            return []
        return list(self.encoder.classes_)
