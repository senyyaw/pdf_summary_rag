# PDF Chatbot (RAG)

A REST API that lets you upload a PDF and ask questions about it. Built with FastAPI and a simple RAG pipeline — no LangChain, just the raw logic.

## How it works

1. You upload a PDF → it gets split into chunks, embedded, and stored in FAISS
2. You ask a question → it finds the most relevant chunks and sends them to an LLM with your question
3. You get an answer based on what's actually in the PDF

## Tech Stack

- **FastAPI** — API framework
- **sentence-transformers** — converts text chunks into vectors (`all-MiniLM-L6-v2`)
- **FAISS** — vector storage and similarity search
- **pypdf** — PDF text extraction
- **Nvidia Nemotron** via OpenRouter — LLM for answering questions

## Getting Started

```bash

git clone https://github.com/senyyaw/RAG_simple.git
cd RAG_simple

python -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# add your api key
cp .env_example .env
# then open .env and paste your OpenRouter key

# run the server
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to use the interactive UI.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF to index |
| POST | `/ask` | Ask a question about the PDF |

## Usage

**Upload a PDF**
```json
POST /upload
// form-data with a PDF file
```

**Ask a question**
```json
POST /ask
{
  "text": "What is this document about?"
}
```

**Response**
```json
{
  "answer": "The document is about...",
  "chunks_used": ["relevant chunk 1", "relevant chunk 2", "relevant chunk 3"]
}
```

## Notes

- Upload a PDF before asking questions
- The model only answers based on the PDF content, not general knowledge
- First startup takes a moment, embedding model loads on boot