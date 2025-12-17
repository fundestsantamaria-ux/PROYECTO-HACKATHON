# app/index.py
import os
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Dict

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-assistant")

def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if INDEX_NAME not in [idx.name for idx in pinecone.list_indexes()]:
        # Dimensión del modelo all-MiniLM-L6-v2 es 384
        pinecone.create_index(INDEX_NAME, dimension=384, metric="cosine")
    return pinecone.Index(INDEX_NAME)

def build_embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_text_for_embedding(article: Dict) -> str:
    title = article.get("title", "")
    body = article.get("body", "")
    return f"Artículo {article['article_number']}: {title}\n{body}"

def upsert_articles(index, model, articles, source_name: str):
    to_upsert = []
    for a in articles:
        text = build_text_for_embedding(a)
        vec = model.encode(text).tolist()

        # 🔹 Guardar solo un fragmento del texto para no superar el límite
        snippet = text[:2000]  # máximo 2000 caracteres (~2 KB)

        metadata = {
            "article_number": a["article_number"],
            "title": a.get("title", ""),
            "text": snippet,   # solo un resumen corto
            "source": source_name,
        }

        to_upsert.append((a["id"], vec, metadata))

    # Subir a Pinecone
    index.upsert(vectors=to_upsert)
    