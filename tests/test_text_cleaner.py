"""
Unit tests for text cleaning and keyword extraction utilities.
"""

import pytest
from src.text_cleaner import clean_resume, extract_skill_keywords


def test_clean_resume_urls_and_mentions():
    raw_text = "Check out https://github.com/Abhi2006-cloud @someone #python RT cc Senior Python Developer"
    cleaned = clean_resume(raw_text)
    assert "https" not in cleaned
    assert "@someone" not in cleaned
    assert "#python" not in cleaned
    assert "RT" not in cleaned
    assert "cc" not in cleaned
    assert "Senior Python Developer" in cleaned


def test_clean_resume_special_chars():
    raw_text = "Data Scientist!!! Skills: Python, Machine Learning & SQL???"
    cleaned = clean_resume(raw_text)
    assert "Data Scientist Skills Python Machine Learning SQL" in cleaned


def test_clean_resume_empty():
    assert clean_resume("") == ""
    assert clean_resume(None) == ""


def test_extract_skill_keywords():
    sample = "Python developer experienced in Python, Django, Flask, Python API development and SQL."
    keywords = extract_skill_keywords(sample, top_n=3)
    assert len(keywords) > 0
    words = [kw[0] for kw in keywords]
    assert "python" in words
