import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from app.models.embeddings import EmbeddingWrapper  # <-- uses Gemini embeddings

load_dotenv()

app = FastAPI(title="LLM Test API", version="0.1.0")


# ===== Request Schemas =====
class GenerateRequest(BaseModel):
    prompt: str


class EmbedRequest(BaseModel):
    text: str


# ===== LLM (Gemini) =====
def _chat_llm():
    """
    Initialize the Gemini chat LLM (gemini-1.5-flash).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Set it in your .env")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # ✅ fast, cost-effective
        google_api_key=api_key,
        temperature=0.7,
    )


@app.post("/generate")
async def generate_text(body: GenerateRequest):
    """
    Generate text response from Gemini model.
    """
    try:
        llm = _chat_llm()
        result = await llm.ainvoke(body.prompt)  # ✅ async version
        return {"response": getattr(result, "content", str(result))}
    except Exception as e:
        return {"error": str(e)}


# ===== Embeddings (Gemini) =====
@app.post("/embed")
async def embed_text(body: EmbedRequest):
    """
    Generate embeddings using Gemini (text-embedding-004).
    Returns:
      - embedding_dim: vector size (should be 1024)
      - preview: first 8 numbers for sanity check
    """
    try:
        emb = EmbeddingWrapper(model="text-embedding-004")
        vec = await emb.embed_text(body.text)  # ✅ async call
        return {
            "embedding_dim": len(vec),
            "preview": vec[:8],  # show first few values
        }
    except Exception as e:
        return {"error": str(e)}
