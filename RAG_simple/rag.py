import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load embedding model once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Global storage
chunks = []
index = None

def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    result = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        result.append(chunk)
    return result

def build_index(pdf_path: str):
    global chunks, index
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

def search(query: str, top_k: int = 3) -> list:
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]