import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import shutil
from rag import build_index, search

load_dotenv()

app = FastAPI(title="PDF Chatbot")

# load the openrouter 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

class Question(BaseModel):
    text: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # save the uploaded pdf temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # build the faiss index from the pdf
    build_index(temp_path)

    # remove the temp file after indexing
    os.remove(temp_path)

    return {"message": f"{file.filename} uploaded and indexed successfully"}


@app.post("/ask")
async def ask_question(question: Question):
    # make sure pdf was uploaded first
    if not search:
        raise HTTPException(status_code=400, detail="upload a pdf first")

    # find the most relevant chunks for the question
    relevant_chunks = search(question.text)
    context = "\n\n".join(relevant_chunks)

    # build the prompt with context and question
    prompt = f"""use the following context to answer the question.
    
context:
{context}

question: {question.text}"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "chunks_used": relevant_chunks
    }