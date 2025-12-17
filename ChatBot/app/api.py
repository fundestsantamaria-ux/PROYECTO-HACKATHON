from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 🔹 Cargar variables de entorno
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agent-assit")

# 🔹 Inicializar FastAPI
app = FastAPI(title="Asistente Legal")

# 🔹 Inicializar Pinecone y modelo de embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Modelo de entrada
class AskBody(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def root():
    return {"message": "Servidor funcionando correctamente"}

def search_query(query: str, top_k: int = 3):
    query_emb = model.encode(query).tolist()
    result = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    return result

def compose_answer(results, user_query: str):
    if not results.matches:
        return "No se encontró información relevante para tu consulta."

    best = results.matches[0].metadata
    citation = f"Artículo {best.get('article_number', 'N/A')} — {best.get('source', 'Ley cargada')}"
    snippet = best.get("text", "")

    if len(snippet) > 1200:
        snippet = snippet[:1200] + "..."

    return (
        f"Según la normativa:\n\n"
        f"{snippet}\n\n"
        f"Cita: {citation}\n"
        f"(Consulta: \"{user_query}\")"
    )

@app.post("/ask")
def ask(body: AskBody):
    results = search_query(body.question, body.top_k)
    answer = compose_answer(results, body.question)
    return {
        "answer": answer,
        "results": [m.metadata for m in results.matches]
    }