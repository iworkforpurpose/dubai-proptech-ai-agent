"""
Centralized Google Gemini embedding module with automatic fallback across models.
Models:
  - Primary: models/gemini-embedding-001 (3072 dimensions)
  - Secondary fallback: models/gemini-embedding-2 (3072 dimensions)
Both produce 3072-dimensional vectors matching our PostgreSQL schema.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PRIMARY_MODEL = "models/gemini-embedding-001"
FALLBACK_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIM = 3072

_primary_client = None
_fallback_client = None


def _get_client(model: str) -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set in .env.")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def get_embedding(text: str) -> list[float]:
    """Embed query with automatic fallback between 3072-dim models."""
    global _primary_client, _fallback_client
    if _primary_client is None:
        _primary_client = _get_client(PRIMARY_MODEL)

    try:
        return _primary_client.embed_query(text)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            if _fallback_client is None:
                _fallback_client = _get_client(FALLBACK_MODEL)
            return _fallback_client.embed_query(text)
        raise e


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed documents with automatic fallback between 3072-dim models."""
    global _primary_client, _fallback_client
    if _primary_client is None:
        _primary_client = _get_client(PRIMARY_MODEL)

    try:
        return _primary_client.embed_documents(texts)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            if _fallback_client is None:
                _fallback_client = _get_client(FALLBACK_MODEL)
            return _fallback_client.embed_documents(texts)
        raise e
