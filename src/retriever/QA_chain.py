from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import logging
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from vectorstore.vectordb import vector_database
from document.document_loader import document_loader
from document.text_splitter import text_splitter
from llm.model import get_llm

logger = logging.getLogger(__name__)


def retriever(file):
    """
    Build a retriever from an uploaded PDF file.

    Pipeline: load PDF -> split text -> embed into vector store -> retriever.

    Args:
        file: A Gradio file object or file-path string pointing to a PDF.

    Returns:
        VectorStoreRetriever: A retriever backed by the ChromaDB index.

    Raises:
        ValueError / FileNotFoundError / RuntimeError:
            Propagated from document_loader, text_splitter, or vector_database
            if the file is invalid or processing fails.
    """
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


def retriever_qa(file, query):
    """
    End-to-end question-answering over an uploaded PDF.

    Args:
        file: A Gradio file object or file-path string pointing to a PDF.
        query: The natural-language question to answer.

    Returns:
        str: The LLM-generated answer, or a user-friendly error message
             if something goes wrong.
    """
    # input validation 
    if file is None:
        return "Please upload a PDF document before asking a question."

    if not query or not query.strip():
        return "Please enter a question about the document."

    query = query.strip()

    if len(query) > 2000:
        return (
            "Your question is too long (max 2 000 characters). "
            "Please shorten it and try again."
        )

    # RAG pipeline
    try:
        llm = get_llm()
        retriever_obj = retriever(file)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="refine",
            retriever=retriever_obj,
            return_source_documents=True,
        )

        logger.info("Running QA chain for query: '%s'", query[:80])
        response = qa.invoke({"query": query})
        return response["result"]

    except (ValueError, FileNotFoundError) as exc:
        # User-correctable problems (bad file, empty doc, etc.)
        logger.warning("Validation error in QA pipeline: %s", exc)
        return f"Input error: {exc}"

    except ConnectionError as exc:
        logger.error("LLM connection error: %s", exc)
        return (
            "Could not connect to the Ollama server. "
            "Please make sure Ollama is running at the configured URL."
        )

    except Exception as exc:
        logger.exception("Unexpected error in QA pipeline.")
        return (
            f"An unexpected error occurred: {exc}\n\n"
            "Please check the logs or try again."
        )

