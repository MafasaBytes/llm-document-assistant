from langchain_community.vectorstores import Chroma
import hashlib
import logging
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from embedding.embed_model import get_embedding_model

logger = logging.getLogger(__name__)

_vectordb_cache: dict[str, Chroma] = {}


def _compute_chunks_hash(chunks) -> str:
    """
    Compute a SHA-256 hash over document chunks so we can tell whether
    the vector store needs to be rebuilt.
    """
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.page_content[:200].encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def vector_database(chunks):
    """
    Create or return a cached in-memory ChromaDB vector store.

    Each unique document (identified by a content hash) gets its own
    isolated store. No data is persisted to disk, so uploading a new
    PDF never retrieves stale results from a previous document.

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

    # Pure in-memory — no persist_directory
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
    )

    _vectordb_cache[chunks_hash] = vectordb
    logger.info("Vector store cached successfully.")
    return vectordb
