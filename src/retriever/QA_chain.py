from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import logging
import time
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

# Metadata QA — returns answer + sources + per-step timing

def retriever_qa_with_metadata(file, query):
    """
    End-to-end QA with per-step timing, source attribution, and metrics.

    Args:
        file: A Gradio file path string or file object pointing to a PDF.
        query: The natural-language question to answer.

    Returns:
        tuple: (answer_text, source_excerpts, metrics_dict)
            - answer_text (str): The LLM-generated answer or error message.
            - source_excerpts (list[dict]): Each dict has 'page' and 'excerpt'.
            - metrics_dict (dict): Timing and count metrics for the pipeline.
    """
    metrics = {}

    # coerce query to str (Gradio 5.x may pass a list)
    if not isinstance(query, str):
        query = str(query)

    # input validation
    if file is None:
        return ("Please upload a PDF document before asking a question.", [], metrics)

    if not query or not query.strip():
        return ("Please enter a question about the document.", [], metrics)

    query = query.strip()

    if len(query) > 2000:
        return ("Your question is too long (max 2,000 characters).", [], metrics)

    # instrumented RAG pipeline 
    try:
        t_start = time.perf_counter()

        # Load PDF
        t0 = time.perf_counter()
        splits = document_loader(file)
        metrics["load_time"] = round(time.perf_counter() - t0, 2)
        metrics["num_pages"] = len(splits)

        # Chunk text
        t0 = time.perf_counter()
        chunks = text_splitter(splits)
        metrics["chunk_time"] = round(time.perf_counter() - t0, 2)
        metrics["num_chunks"] = len(chunks)

        # Embed & index into vector store
        t0 = time.perf_counter()
        vectordb = vector_database(chunks)
        metrics["embed_time"] = round(time.perf_counter() - t0, 2)

        # Retrieve & generate answer
        t0 = time.perf_counter()
        llm = get_llm()
        retriever_obj = vectordb.as_retriever()

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="refine",
            retriever=retriever_obj,
            return_source_documents=True,
        )

        logger.info("Running QA chain for query: '%s'", query[:80])
        response = qa.invoke({"query": query})
        metrics["generation_time"] = round(time.perf_counter() - t0, 2)
        metrics["total_latency"] = round(time.perf_counter() - t_start, 2)

        # Extract source documents
        source_docs = response.get("source_documents", [])
        metrics["num_sources"] = len(source_docs)

        sources = []
        seen_pages = set()
        for doc in source_docs:
            page = doc.metadata.get("page", "?")
            if page in seen_pages:
                continue
            seen_pages.add(page)
            sources.append({
                "page": page,
                "excerpt": doc.page_content[:200].strip(),
            })

        return (response["result"], sources, metrics)

    except (ValueError, FileNotFoundError) as exc:
        logger.warning("Validation error: %s", exc)
        return (f"Input error: {exc}", [], metrics)

    except ConnectionError as exc:
        logger.error("Connection error: %s", exc)
        return (
            "Could not connect to the Ollama server. "
            "Please ensure Ollama is running.",
            [],
            metrics,
        )

    except Exception as exc:
        logger.exception("Unexpected error in QA pipeline.")
        return (f"An unexpected error occurred: {exc}", [], metrics)
