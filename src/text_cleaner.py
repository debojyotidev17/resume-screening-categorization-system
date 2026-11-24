"""
Text cleaning and NLP utility functions for resume preprocessing.
"""

import re
import string
from collections import Counter
from typing import List, Tuple

# Common stop words set for keyword extraction filtering
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "experience", "work", "project", "using", "used", "skills", "knowledge", "ability"
}


def clean_resume(txt: str) -> str:
    """
    Cleans raw resume text by stripping URLs, hashtags, mentions,
    punctuations, special characters, and extraneous whitespace.

    Args:
        txt (str): Raw extracted resume text.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(txt, str):
        return ""

    # Remove URLs
    clean_text = re.sub(r'http\S+\s*', ' ', txt)
    # Remove RT and cc
    clean_text = re.sub(r'\b(RT|cc)\b', ' ', clean_text)
    # Remove hashtags and mentions
    clean_text = re.sub(r'#\S+', ' ', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    # Remove punctuation
    clean_text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', clean_text)
    # Remove non-ASCII characters
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    # Collapse multiple whitespaces
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text.strip()


def extract_skill_keywords(txt: str, top_n: int = 15) -> List[Tuple[str, int]]:
    """
    Extracts top skill keywords and frequencies from resume text for insights dashboard.

    Args:
        txt (str): Input resume text.
        top_n (int): Number of top keywords to retrieve.

    Returns:
        List[Tuple[str, int]]: List of (word, count) tuples.
    """
    if not txt:
        return []

    words = re.findall(r'\b[a-zA-Z]{3,}\b', txt.lower())
    filtered_words = [w for w in words if w not in STOP_WORDS]
    counts = Counter(filtered_words)
    return counts.most_common(top_n)
