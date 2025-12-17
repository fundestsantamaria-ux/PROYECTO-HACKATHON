# create_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-assistant")

# 🔹 Verifica que la API key esté cargada
if not PINECONE_API_KEY:
    raise ValueError("❌ No se encontró PINECONE_API_KEY en el entorno.")

# 🔹 Inicializa cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# 🔹 Crea el índice si no existe
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # para all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="awx      s",         # usa "aws" si tu cuenta está en AWS
            region="us-east-1"   # asegúrate que coincida con tu entorno
        )
    )
    print(f"✅ Índice '{INDEX_NAME}' creado correctamente.")
else:
    print(f"ℹ️ El índice '{INDEX_NAME}' ya existe.")