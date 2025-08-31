# app/llm.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY is missing. Add it to your .env file.")

# Default model for Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def get_llm(temperature: float = 0.7, max_output_tokens: int = 1024):
    """
    Returns a LangChain ChatGoogleGenerativeAI instance.
    """
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def generate_text(prompt: str, temperature: float = 0.7) -> str:
    """
    Simple helper to generate text from Gemini.
    """
    llm = get_llm(temperature=temperature)
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)
