"""
Centralized embedding module.
Reason for choice: text-embedding-3-large (OpenAI)
  - 3072 dimensions (highest fidelity)
  - MTEB score: 64.6 (best available via API)
  - Sub-300ms latency via API vs 8-12s on local CPU
  - Zero local model management — scales automatically
"""
import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072


def get_embedding(text: str) -> list[float]:
    """Embed a single text string using OpenAI text-embedding-3-large."""
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one API call (more efficient for ingestion)."""
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]
