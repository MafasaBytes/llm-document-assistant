from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def document_loader(input_file):
    """
    Load a PDF file and return a list of LangChain Document objects.

    Accepts either a file-like object with a `.name` attribute (as
    provided by Gradio's File component) or a plain file-path string.

    Args:
        input_file: A Gradio TemporaryFile / NamedString or a str / Path
                     pointing to the PDF on disk.

    Returns:
        list[Document]: Parsed pages of the PDF.

    Raises:
        ValueError: If no file is provided.
        FileNotFoundError: If the resolved path does not exist.
        ValueError: If the file is not a PDF.
        RuntimeError: If the PDF cannot be parsed.
    """
    # resolve the file path
    if input_file is None:
        raise ValueError("No file provided. Please upload a PDF document.")

    # Gradio may pass a TemporaryFile object or a plain path string
    if hasattr(input_file, "name"):
        file_path = Path(input_file.name)
    else:
        file_path = Path(str(input_file))

    # validate 
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file but received '{file_path.suffix}'. "
            "Please upload a valid PDF document."
        )

    if file_path.stat().st_size == 0:
        raise ValueError("The uploaded PDF file is empty (0 bytes).")

    # load 
    logger.info("Loading PDF: %s", file_path)

    try:
        loader = PyPDFLoader(str(file_path))
        loaded_document = loader.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse the PDF file: {exc}"
        ) from exc

    if not loaded_document:
        raise RuntimeError(
            "The PDF was loaded but produced no pages. "
            "It may be scanned/image-only or corrupted."
        )

    logger.info("Loaded %d page(s) from PDF.", len(loaded_document))
    return loaded_document