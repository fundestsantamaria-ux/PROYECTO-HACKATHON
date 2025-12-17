# run_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from app.ingest import load_legal_articles
from app.index import build_embeddings_model, upsert_articles

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-assistant")

# Inicializar cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Crear índice si no existe
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,   # dimensión del modelo all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",       # o "gcp"
            region="us-east-1" # ajusta según tu cuenta
        )
    )
    print(f"✅ Índice '{INDEX_NAME}' creado en Pinecone.")

# Conectar al índice
index = pc.Index(INDEX_NAME)

# Ruta al PDF legal
PDF_PATH = "data/ley_base.pdf"
SOURCE_NAME = "Ley Base (PDF local)"

if __name__ == "__main__":
    articles = load_legal_articles(PDF_PATH)
    print(f"Artículos listos para indexar: {len(articles)}")
    model = build_embeddings_model()
    upsert_articles(index, model, articles, SOURCE_NAME)
    print("✅ Indexación completada en Pinecone.")