# app/ingest.py
from PyPDF2 import PdfReader
import re
from typing import List, Dict

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def normalize_text(text: str) -> str:
    # Quita espacios repetidos, normaliza saltos de línea
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def split_by_articles(text: str) -> List[Dict]:
    """
    Divide por 'Artículo' capturando número y título cuando sea posible.
    Compatible con formatos: 'Artículo 15.-', 'Art. 15', etc.
    """
    pattern = r'(?:\n|^)(Artículo|Art\.)\s+(\d+)[\.\-–:]?\s*(.*?)(?=\n(?:Artículo|Art\.)\s+\d+|$)'
    matches = re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    articles = []
    for m in matches:
        label = m.group(1)
        number = m.group(2)
        maybe_title = m.group(3).strip()
        content = maybe_title

        parts = content.split("\n", 1)
        title = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""

        articles.append({
            "id": f"art_{number}",
            "article_number": number,
            "title": title,
            "body": body if body else title,
        })
    return articles

def load_legal_articles(pdf_path: str) -> List[Dict]:
    raw = extract_text_from_pdf(pdf_path)
    norm = normalize_text(raw)
    articles = split_by_articles(norm)
    return articles