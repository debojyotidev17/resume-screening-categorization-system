"""
File parsing utilities for extracting raw text from PDF, DOCX, and TXT resumes.
"""

import io
from typing import Union, BinaryIO
import docx
import PyPDF2


def extract_text_from_pdf(file_source: Union[str, BinaryIO, io.BytesIO]) -> str:
    """
    Extracts text contents from a PDF file source.

    Args:
        file_source: File path or binary file-like stream.

    Returns:
        str: Extracted text from all pages.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file_source)
        text_pages = []
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_pages.append(extracted)
        return "\n".join(text_pages)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}") from e


def extract_text_from_docx(file_source: Union[str, BinaryIO, io.BytesIO]) -> str:
    """
    Extracts text contents from a Microsoft Word (.docx) file source.

    Args:
        file_source: File path or binary file-like stream.

    Returns:
        str: Extracted paragraph text.
    """
    try:
        doc = docx.Document(file_source)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}") from e


def extract_text_from_txt(file_source: Union[str, BinaryIO, io.BytesIO]) -> str:
    """
    Extracts text contents from a plain text (.txt) file source with encoding fallbacks.

    Args:
        file_source: File path or binary file-like stream.

    Returns:
        str: Decoded plain text content.
    """
    try:
        if isinstance(file_source, str):
            with open(file_source, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        # Handle binary stream / UploadedFile
        raw_bytes = file_source.read()
        # Reset stream position if possible
        if hasattr(file_source, 'seek'):
            file_source.seek(0)

        try:
            return raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return raw_bytes.decode('latin-1', errors='ignore')

    except Exception as e:
        raise ValueError(f"Failed to extract text from TXT: {str(e)}") from e


def parse_resume_file(file_source: Union[str, BinaryIO, io.BytesIO], filename: str = "") -> str:
    """
    Auto-detects file extension and extracts raw text.

    Args:
        file_source: File path or binary file-like object.
        filename (str): Name of the file (used to determine extension if file_source is a stream).

    Returns:
        str: Raw text extracted from resume.
    """
    name = filename if filename else (file_source if isinstance(file_source, str) else getattr(file_source, 'name', ''))
    ext = name.split('.')[-1].lower() if '.' in name else ''

    if ext == 'pdf':
        return extract_text_from_pdf(file_source)
    elif ext == 'docx':
        return extract_text_from_docx(file_source)
    elif ext == 'txt' or not ext:
        return extract_text_from_txt(file_source)
    else:
        raise ValueError(f"Unsupported file extension '.{ext}'. Supported formats: PDF, DOCX, TXT.")
