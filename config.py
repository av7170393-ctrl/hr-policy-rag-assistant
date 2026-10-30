"""Central configuration. Override anything via environment variables or .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for tests
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "policies"
ACCESS_FILE = BASE_DIR / "data" / "access.json"
STORE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORE_DIR / "chroma"
CHUNKS_FILE = STORE_DIR / "chunks.json"
DB_PATH = Path(os.getenv("DB_PATH", STORE_DIR / "hr.db"))

# Multilingual embeddings let Hindi / Hinglish questions match English policies
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# LLM provider: "gemini" (free Google AI Studio key) or "groq". Both speak the OpenAI API format.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
}
LLM_MODEL = os.getenv("LLM_MODEL", PROVIDERS.get(LLM_PROVIDER, PROVIDERS["gemini"])["model"])
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")  # used only with the groq provider

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "12"))

TOP_K_DENSE = 10
TOP_K_BM25 = 10
TOP_K_FUSED = 8
TOP_K_FINAL = 4
# Cross-encoder score below which we say "not in the policies" and escalate to HR
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))

ROLE_LEVEL = {"employee": 1, "manager": 2, "hr": 3}
