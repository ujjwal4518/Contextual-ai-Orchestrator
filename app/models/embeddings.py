import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class EmbeddingWrapper:
    """
    Thin wrapper around Google (Gemini) embeddings for both single text
    and batch documents. Uses text-embedding-004 (1024 dims).
    """

    def __init__(self, model: str = "text-embedding-004", api_key: str | None = None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Set it in your .env")
        self.model_name = model
        self._emb = GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=api_key)

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single string asynchronously. Returns 1024-dim vector.
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        return await self._emb.aembed_query(text)

    async def embed_documents(self, docs: List[str]) -> List[List[float]]:
        """
        Embed multiple strings asynchronously. Each row is a 1024-dim vector.
        """
        if not docs:
            raise ValueError("docs cannot be empty")
        return await self._emb.aembed_documents(docs)
