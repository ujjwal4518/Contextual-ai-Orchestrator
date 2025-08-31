import os
from datetime import timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
import uvicorn

# ==== Internal Imports ====
from app.security.auth import authenticate_user, create_access_token, get_current_user
from app.models.embeddings import EmbeddingWrapper
from app.rag.ingest import prepare_chunks
from app.rag.vectorstore import (
    add_documents_to_store, 
    search_similar_docs, 
    list_collections, 
    get_collection_info
)
from langchain_google_genai import ChatGoogleGenerativeAI

# ==== Load environment ====
load_dotenv()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==== FastAPI App ====
app = FastAPI(title="Contextual AI Orchestrator", version="1.0.0")

# ==== Auth Settings ====
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ===== Schemas =====
class IngestRequest(BaseModel):
    file_id: str

class AskRequest(BaseModel):
    question: str
    file_id: str
    top_k: int = 4

class GenerateRequest(BaseModel):
    prompt: str

class EmbedRequest(BaseModel):
    text: str

# ===== Utility Functions =====
def _chat_llm():
    """
    Initialize Gemini chat LLM.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Set it in your .env")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )

# ===== Core Endpoints =====

@app.get("/")
async def root():
    return {"message": "Contextual AI Orchestrator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# ==== Authentication Routes ====
@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = authenticate_user(form_data.username, form_data.password)
    if not username:
        return {"error": "Invalid credentials"}
    token = create_access_token(
        {"sub": username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer"}

@app.get("/protected")
def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": f"Hello, {current_user}. You have access to this protected route!"}

# ==== Upload & Ingest ====
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_id = os.path.splitext(file.filename)[0]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "message": "File uploaded successfully",
        "file_id": file_id,
        "file_size_bytes": len(content)
    }

@app.post("/ingest")
async def ingest_file(body: IngestRequest):
    chunks = prepare_chunks(body.file_id)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No text found in {body.file_id}.pdf")

    count = add_documents_to_store(chunks, body.file_id)
    collection_info = get_collection_info(body.file_id)

    return {
        "message": f"File {body.file_id} processed successfully",
        "chunks_added": count,
        "collection_info": collection_info
    }

# ==== Ask Question ====
@app.post("/ask")
async def ask_question(body: AskRequest):
    docs = search_similar_docs(body.question, body.file_id, k=body.top_k)
    if not docs:
        return {
            "answer": "No relevant information found in the document.",
            "sources": []
        }

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {body.question}\nAnswer:"

    llm = _chat_llm()
    result = await llm.ainvoke(prompt)

    return {
        "answer": getattr(result, "content", str(result)),
        "sources": [doc.metadata for doc in docs],
        "context_length": len(context),
        "num_results": len(docs)
    }

# ==== Debug & Collections ====
@app.get("/collections")
async def list_all_collections():
    return {"collections": list_collections(), "total_count": len(list_collections())}

@app.get("/collections/{collection_id}")
async def get_collection_details(collection_id: str):
    return get_collection_info(collection_id)

@app.get("/debug/uploads")
async def list_uploaded_files():
    if not os.path.exists(UPLOAD_DIR):
        return {"files": [], "upload_dir": UPLOAD_DIR, "exists": False}

    files = [
        {
            "filename": filename,
            "size_bytes": os.path.getsize(os.path.join(UPLOAD_DIR, filename)),
            "file_id": os.path.splitext(filename)[0]
        }
        for filename in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, filename))
    ]
    return {"files": files, "total_count": len(files), "upload_dir": UPLOAD_DIR, "exists": True}

# ==== LLM Test Endpoints ====
@app.post("/generate")
async def generate_text(body: GenerateRequest):
    llm = _chat_llm()
    result = await llm.ainvoke(body.prompt)
    return {"response": getattr(result, "content", str(result))}

@app.post("/embed")
async def embed_text(body: EmbedRequest):
    emb = EmbeddingWrapper(model="text-embedding-004")
    vec = await emb.embed_text(body.text)
    return {"embedding_dim": len(vec), "preview": vec[:8]}

# ==== Run Application ====
if __name__ == "__main__":
    uvicorn.run("app1:app", host="127.0.0.1", port=8080, reload=True)
