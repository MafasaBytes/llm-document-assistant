from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def text_splitter(data, chunk_size=1000, chunk_overlap=100):
    """
    Split a list of LangChain Documents into smaller text chunks.

    Args:
        data: List of Document objects (typically from document_loader).
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        list[Document]: Non-empty text chunks ready for embedding.

    Raises:
        ValueError: If the input data is empty or produces no usable chunks.
    """
    if not data:
        raise ValueError(
            "No documents provided to the text splitter. "
            "The PDF may not have contained any extractable text."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(data)

    # Filter out empty / whitespace-only chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError(
            "Text splitting produced no usable chunks. "
            "The document may contain only images or blank pages."
        )

    logger.info("Split documents into %d non-empty chunks.", len(chunks))
    return chunks