"""
Unit tests for resume document parser functionality.
"""

import io
import pytest
from src.file_parser import parse_resume_file, extract_text_from_txt


def test_extract_text_from_txt_stream():
    sample_content = "Abhi Suryawanshi\nSoftware Engineer\nSkills: Python, Docker, React"
    bytes_stream = io.BytesIO(sample_content.encode('utf-8'))
    extracted = extract_text_from_txt(bytes_stream)
    assert "Abhi Suryawanshi" in extracted
    assert "Software Engineer" in extracted


def test_parse_resume_file_txt():
    sample_content = "Network Security Engineer with 5 years experience in Firewalls and Cisco."
    bytes_stream = io.BytesIO(sample_content.encode('utf-8'))
    extracted = parse_resume_file(bytes_stream, filename="resume.txt")
    assert "Network Security Engineer" in extracted


def test_parse_resume_unsupported_format():
    bytes_stream = io.BytesIO(b"dummy binary data")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        parse_resume_file(bytes_stream, filename="resume.xyz")
