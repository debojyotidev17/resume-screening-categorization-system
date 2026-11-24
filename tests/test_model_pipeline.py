"""
Unit tests for ResumeClassifier model pipeline.
"""

import os
import pytest
from src.model_pipeline import ResumeClassifier


def test_resume_classifier_initialization():
    classifier = ResumeClassifier()
    # Check if classifier initializes without errors
    assert hasattr(classifier, 'is_loaded')


def test_resume_classifier_prediction_when_loaded():
    classifier = ResumeClassifier()
    if classifier.is_loaded:
        sample_resume = "Experienced Data Scientist skilled in Python, Machine Learning, Deep Learning, Pandas, Scikit-learn, TensorFlow."
        pred_category = classifier.predict(sample_resume)
        assert isinstance(pred_category, str)
        assert len(pred_category) > 0

        top_k = classifier.predict_top_k(sample_resume, top_k=3)
        assert len(top_k) == 3
        assert isinstance(top_k[0][0], str)
        assert isinstance(top_k[0][1], float)
