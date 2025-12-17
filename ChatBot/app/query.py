# app/query.py
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import pinecone

class LegalSearcher:
    def __init__(self, index: pinecone.Index, model: SentenceTransformer):
        self.index = index
        self.model = model

    def search(self, user_query: str, top_k: int = 5) -> List[Dict]:
        qvec = self.model.encode(user_query).tolist()
        res = self.index.query(vector=qvec, top_k=top_k, include_metadata=True)
        return [
            {
                "id": m.id,
                "score": m.score,
                "article_number": m.metadata.get("article_number"),
                "title": m.metadata.get("title"),
                "text": m.metadata.get("text"),
                "source": m.metadata.get("source"),
            }
            for m in res.matches
        ]

def compose_answer(results: List[Dict], user_query: str) -> str:
    if not results:
        return "No se encontró información relevante para tu consulta. Intenta reformular la pregunta con más detalles."

    best = results[0]
    citation = f"Artículo {best['article_number']} — {best['source']}"
    snippet = best["text"]
    if len(snippet) > 1200:
        snippet = snippet[:1200] + "..."

    return (
        f"Según la normativa aplicable:\n\n"
        f"{snippet}\n\n"
        f"Cita: {citation}\n"
        f"(Consulta: \"{user_query}\")"
    )