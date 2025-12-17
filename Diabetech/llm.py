import os
from typing import List, Dict, Optional
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
# -----------------------------

# ---------------- Config ----------------
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
CHROMA_DB_BASE_DIR = "./vectors"
CHROROMA_COLLECTIONS = ["first", "second"] 
EMBEDDING_MODEL = "nomic-embed-text:latest"
# -----------------------------

DEFAULT_MODEL = OLLAMA_MODEL

# Inicializar cliente Ollama
try:
    _client = ollama.Client(host=OLLAMA_API_BASE)
    _init_error = None
except Exception as e:
    _client = None
    _init_error = str(e)

_vector_dbs: List[Chroma] = []
_db_loaded = False
_db_init_messages = []

try:
    # 1. Inicializar el modelo de Embeddings usado para la DB
    _embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL, 
        base_url=OLLAMA_API_BASE
    )

    # 2. Cargar MÚLTIPLES bases de datos Chroma
    for collection_name in CHROROMA_COLLECTIONS:
        db_path = os.path.join(CHROMA_DB_BASE_DIR, collection_name)
        db = Chroma(
            persist_directory=db_path, 
            embedding_function=_embeddings
        )
        _vector_dbs.append(db)
        _db_init_messages.append(f"Cargada colección: {collection_name}")
    
    _db_loaded = True
    print("\n".join(_db_init_messages))

except Exception as e:
    _db_loaded = False
    print(f"Advertencia: No se pudo cargar una o más bases de datos Chroma. El RAG estará deshabilitado. Error: {e}")

SYSTEM_PROMPT = """\
Eres "DiabeTech Assistant", un asistente clínico–económico experto.

Tu misión es ayudar a jurados y usuarios no técnicos a:
- Entender un sistema de IA para predicción de riesgo de diabetes
- Interactuar con un modelo de Aprendizaje Federado
- Visualizar impacto social y económico en un contexto de recuperación post-pandemia

=====================================================
CONTEXTO DEL PROYECTO
=====================================================
- El sistema utiliza Aprendizaje Federado (Federated Learning) para entrenar un modelo MLP de predicción de riesgo de diabetes.
- Se tiene la arquitectura Centralizada: un servidor fijo agrega los modelos locales.
- Los nodos representan hospitales o centros de salud.
- Los datos clínicos NUNCA se centralizan: solo se intercambian parámetros del modelo.
- La comunicación entre nodos se realiza mediante TCP sockets y el despliegue se orquesta con Docker Compose.

=====================================================
OBJETIVOS DEL ASISTENTE
=====================================================
1) Explicar qué significa cada variable del formulario y qué valores ingresar (0/1, rangos, categorías).
2) Explicar Federated Learning, privacidad de datos, rotación de líder y beneficios institucionales.
3) Traducir resultados técnicos en impacto social y económico entendible.
4) Servir como herramienta de apoyo para un pitch de hackathon.

=====================================================
REGLAS DE COMPORTAMIENTO
=====================================================
- Responde SIEMPRE en español claro, profesional y orientado a demo/pitch.
- NO realices diagnósticos médicos.
- Incluye un disclaimer breve cuando hables de riesgo:
  “Esto es informativo y no reemplaza la evaluación de un profesional de la salud.”
- No inventes métricas del modelo (accuracy, F1, etc.). Si no existen, dilo explícitamente.
- Si faltan datos, pregunta solo lo estrictamente necesario.
- Mantén respuestas concisas por defecto; amplía solo si el usuario lo solicita.

=====================================================
VARIABLES DEL DATASET (FUENTE REAL)
=====================================================

VARIABLES BINARIAS (0 = No, 1 = Sí):

- HighBP: Diagnóstico de presión arterial alta (1=Sí, 0=No)
- HighChol: Diagnóstico de colesterol alto (1=Sí, 0=No)
- CholCheck: Chequeo de colesterol en los últimos 5 años (1=Sí, 0=No)
- Smoker: Ha fumado ≥100 cigarrillos en su vida (1=Sí, 0=No)
- Stroke: Diagnóstico previo de derrame cerebral (1=Sí, 0=No)
- HeartDiseaseorAttack: Enfermedad coronaria o infarto (1=Sí, 0=No)
- PhysActivity: Actividad física en últimos 30 días (1=Sí, 0=No)
- Fruits: Consume fruta al menos una vez al día (1=Sí, 0=No)
- Veggies: Consume vegetales al menos una vez al día (1=Sí, 0=No)
- HvyAlcoholConsump: Consumo excesivo de alcohol (1=Sí, 0=No)
  Definición: Hombres >14 tragos/semana, Mujeres >7 tragos/semana
- AnyHealthcare: Tiene cobertura de salud (1=Sí, 0=No)
- NoDocbcCost: No pudo ver a un médico por costo (1=Sí, 0=No)
- DiffWalk: Dificultad seria para caminar/subir escaleras (1=Sí, 0=No)
- Sex: Sexo biológico para el modelo (1=Hombre, 0=Mujer)

VARIABLES CONTINUAS / ENTERAS
- BMI: Índice de Masa Corporal (sugerido 10–70)
- MentHlth: Días (0–30) con mala salud mental en últimos 30 días
- PhysHlth: Días (0–30) con mala salud física en últimos 30 días

VARIABLES ORDINALES / CATEGÓRICAS
- GenHlth: 1=Excelente, 2=Muy buena, 3=Buena, 4=Regular, 5=Mala
- Age: Categoría 1–14 (1=18–24 ... 14=80+)
- Education: 1–6
- Income: 1–8

=====================================================
SALIDA ESPERADA (FORMATO DEMO)
=====================================================
Cuando el usuario solicite evaluación del formulario o interpretación:
1) Resumen claro de inputs (si se proporcionan).
2) Interpretación del riesgo (bajo / moderado / elevado) con disclaimer.
3) Recomendaciones generales de prevención (no médicas).
4) Cierre: privacidad, escalabilidad, colaboración inter-hospitalaria.

MENSAJE CLAVE:
“El valor de DiabeTech no es solo predecir riesgo,
sino permitir que múltiples hospitales colaboren,
protegiendo datos sensibles,
reduciendo costos
y mejorando prevención en salud pública.”
"""

def build_input_messages(chat_history: List[Dict[str, str]], mode: str, rag_context: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Construye los mensajes para Ollama, inyectando el contexto RAG en el SYSTEM_PROMPT.
    """
    current_system_prompt = SYSTEM_PROMPT
    
    if rag_context:
        # Añade el contexto RAG al prompt del sistema para el LLM
        current_system_prompt = f"""\
        {current_system_prompt}
        
        =====================================================
        INFORMACIÓN DE REFERENCIA ADICIONAL (DB)
        =====================================================
        Usa la siguiente información si es relevante para la consulta del usuario.
        
        {rag_context}
        """

    msgs: List[Dict[str, str]] = [{"role": "system", "content": current_system_prompt}]

    limit = 12
    tail = chat_history[-limit:] if len(chat_history) > limit else chat_history

    for m in tail:
        role = m.get("role", "")
        content = (m.get("content", "") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})

    return msgs


def llm_reply(
    chat_history: List[Dict[str, str]],
    mode: str,
    model: Optional[str] = None,
) -> str:
    if _client is None:
        return f"Error: no se pudo inicializar Ollama Client con {OLLAMA_API_BASE}. Detalle: {_init_error}"

    model = model or DEFAULT_MODEL
    
    # ---------------- 1. RAG Retrieval (Múltiple) ----------------
    rag_context = None
    
    # Obtenemos la última pregunta del usuario para usarla como query de búsqueda
    last_user_message = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), None)
    
    if _db_loaded and last_user_message:
        all_retrieved_docs = []
        total_docs_retrieved = 0
        try:
            # Iterar sobre todas las bases de datos cargadas
            for db in _vector_dbs:
                # Buscar 2 documentos por cada DB (ajusta el número 'k' si es necesario)
                docs = db.similarity_search(last_user_message, k=2) 
                all_retrieved_docs.extend(docs)
                total_docs_retrieved += len(docs)
            
            # Formatear todos los documentos para inyectarlos en el prompt
            context_texts = []
            for doc in all_retrieved_docs:
                # Usar el nombre de la colección (opcional, requiere ajustes avanzados)
                # O simplemente usar la metadata existente
                context_texts.append(f"Fuente (Pág. {doc.metadata.get('page', 'N/A')}): {doc.page_content}")
                
            rag_context = "\n---\n".join(context_texts)
            print(f"RAG: Se recuperaron {total_docs_retrieved} documentos en total desde {len(_vector_dbs)} fuentes.")

        except Exception as e:
            print(f"Error durante la búsqueda RAG múltiple: {e}")
            rag_context = None # Continúa sin RAG

    # ---------------- 2. Generación con Contexto ----------------
    messages = build_input_messages(chat_history, mode, rag_context)

    try:
        response = _client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.4, "top_p": 0.9, "penalty_score": 1.3},
            keep_alive=0
        )
        return (response.get("message", {}).get("content", "") or "").strip()

    except ollama.ResponseError as e:
        return f"Error de Ollama: {e}. Verifica que el modelo '{model}' esté instalado y corriendo."
    except Exception as e:
        return f"Error inesperado llamando a Ollama: {e}"