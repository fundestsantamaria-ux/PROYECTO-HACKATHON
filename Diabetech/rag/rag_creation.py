from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# --- 1. Cargar PDF ---
loader = PyPDFLoader("spanish-tasty-recipe-508.pdf")
docs = loader.load_and_split()

# --- 2. Crear embeddings ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 3. Crear vector store ---
vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="../vectors/second")
vectordb.persist()

print("Vector store creado y guardado.")