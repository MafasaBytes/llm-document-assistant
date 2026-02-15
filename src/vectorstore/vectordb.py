from langchain_community.vectorstores import Chroma
from pathlib import Path
import hashlib
import logging
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from embedding.embed_model import get_embedding_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vector store cache — one store per unique document set
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

_vectordb_cache: dict[str, Chroma] = {}


def _compute_chunks_hash(chunks) -> str:
    """
    Compute a lightweight hash over document chunks so we can tell whether
    the vector store needs to be rebuilt.

    Uses the first 200 chars of each chunk's page_content to build the hash
    so it's fast even for large documents.
    """
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.page_content[:200].encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def vector_database(chunks):
    """
    Create or return a cached ChromaDB vector store for the given chunks.

    If the same set of chunks has been indexed before (determined by a
    content hash), the existing in-memory vector store is returned
    immediately instead of re-embedding and re-writing to disk.

    Args:
        chunks: List of LangChain Document objects to index.

    Returns:
        Chroma: A ready-to-query ChromaDB vector store.

    Raises:
        ValueError: If the chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty chunk list.")

    chunks_hash = _compute_chunks_hash(chunks)

    # Return cached store if we've already indexed these exact chunks
    if chunks_hash in _vectordb_cache:
        logger.info("Reusing cached vector store (hash=%s...).", chunks_hash[:12])
        return _vectordb_cache[chunks_hash]

    logger.info(
        "Building new vector store for %d chunks (hash=%s...) ...",
        len(chunks),
        chunks_hash[:12],
    )

    embed_model = get_embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    _vectordb_cache[chunks_hash] = vectordb
    logger.info("Vector store cached successfully.")
    return vectordb
