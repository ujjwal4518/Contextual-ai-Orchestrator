import os
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "3600"))

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
VECTOR_DIR = os.getenv("VECTOR_DIR", "./data/vectorstore")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
