# app/main.py (or whatever your main file is)
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
from app.rag.ingest import prepare_chunks
from app.rag.vectorstore import (
    add_documents_to_store, 
    search_similar_docs, 
    list_collections, 
    get_collection_info
)
# from app.models.llm import generate_text  # Make sure this exists

load_dotenv()

app = FastAPI(title="Contextual AI Orchestrator", version="1.0.0")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===== Schemas =====
class IngestRequest(BaseModel):
    file_id: str

class AskRequest(BaseModel):
    question: str
    file_id: str
    top_k: int = 4

# ===== Endpoints =====

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file to the uploads folder.
    Returns file_id for further operations.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_id = os.path.splitext(file.filename)[0]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
        
        # Read and save file
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"✅ File uploaded: {file_id}.pdf ({len(content)} bytes)")
        return {
            "message": "File uploaded successfully", 
            "file_id": file_id,
            "file_size_bytes": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ingest")
async def ingest_file(body: IngestRequest):
    """
    Process uploaded PDF and create vector embeddings
    """
    try:
        print(f"🔄 Starting ingestion for file_id: {body.file_id}")
        
        # Prepare chunks
        chunks = prepare_chunks(body.file_id)
        
        if not chunks:
            raise HTTPException(status_code=400, detail=f"No text found in {body.file_id}.pdf")
        
        print(f"🔄 Processing {len(chunks)} chunks for ingestion")
        
        # Add to vector store
        count = add_documents_to_store(chunks, body.file_id)
        
        # Get collection info for verification
        collection_info = get_collection_info(body.file_id)
        
        print(f"✅ Ingestion completed: {count} chunks processed")
        return {
            "message": f"File {body.file_id} processed successfully", 
            "chunks_added": count,
            "collection_info": collection_info
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/ask")
async def ask_question(body: AskRequest):
    """
    Search the knowledge base and generate an answer.
    """
    try:
        print(f"🔄 Processing question: '{body.question}' for file_id: {body.file_id}")
        
        # Search for relevant documents
        docs = search_similar_docs(body.question, body.file_id, k=body.top_k)
        
        if not docs:
            return {
                "answer": "No relevant information found in the document.",
                "sources": [],
                "debug_info": {
                    "collection_info": get_collection_info(body.file_id)
                }
            }
        
        # Build context from top chunks
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {body.question}\nAnswer:"
        
        # For now, return context (implement generate_text later)
        answer = f"Based on the document content: {context[:500]}..."
        
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in docs],
            "context_length": len(context),
            "num_results": len(docs)
        }
        
    except ValueError as e:
        # Add debugging info to the error response
        debug_info = {
            "available_collections": list_collections(),
            "requested_collection": body.file_id,
            "collection_info": get_collection_info(body.file_id)
        }
        print(f"❌ ValueError: {str(e)}")
        print(f"❌ Debug info: {debug_info}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# ===== New debugging endpoints =====

@app.get("/collections")
async def list_all_collections():
    """
    List all available collections in the vector store.
    """
    try:
        collections = list_collections()
        return {
            "collections": collections,
            "total_count": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.get("/collections/{collection_id}")
async def get_collection_details(collection_id: str):
    """
    Get detailed information about a specific collection.
    """
    try:
        info = get_collection_info(collection_id)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")

@app.get("/debug/uploads")
async def list_uploaded_files():
    """
    List all files in the uploads directory.
    """
    try:
        if not os.path.exists(UPLOAD_DIR):
            return {"files": [], "upload_dir": UPLOAD_DIR, "exists": False}
        
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(filepath):
                files.append({
                    "filename": filename,
                    "size_bytes": os.path.getsize(filepath),
                    "file_id": os.path.splitext(filename)[0]
                })
        
        return {
            "files": files,
            "total_count": len(files),
            "upload_dir": UPLOAD_DIR,
            "exists": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing uploads: {str(e)}")

# ===== Original endpoints =====

@app.get("/")
async def root():
    return {"message": "Contextual AI Orchestrator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}