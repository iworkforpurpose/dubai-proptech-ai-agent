"""
Centralized Google Gemini embedding module.
Model: text-embedding-004
  - 768 dimensions (optimal for fast vector indexing in pgvector)
  - Top tier on MTEB Retrieval benchmark
  - Free tier available on Google AI Studio
  - Cloud API, sub-300ms latency, zero local CPU load
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768

_client = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. "
                "Please add GEMINI_API_KEY to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_embedding(text: str) -> list[float]:
    """Embed a single text string using Google text-embedding-004 (768 dimensions)."""
    client = get_genai_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return response.embeddings[0].values


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one call using Google text-embedding-004."""
    client = get_genai_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in response.embeddings]
