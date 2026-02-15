from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import logging
import yaml
import os

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace token validation
# ---------------------------------------------------------------------------

def load_hf_token():
    """
    Load and validate the HuggingFace API token from the .env file.

    Raises:
        EnvironmentError: If HF_TOKEN is not set in the environment.
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise EnvironmentError(
            "HF_TOKEN environment variable not found. "
            "Please add it to your .env file."
        )
    return hf_token

# ---------------------------------------------------------------------------
# Lazy-loaded singleton for the embedding model
# ---------------------------------------------------------------------------

_embed_model_instance = None


def get_embedding_model():
    """
    Return a lazily-initialized singleton of the HuggingFace embedding model.

    The model is created on the first call and reused for all subsequent
    calls, avoiding the overhead of re-downloading / re-loading weights
    every time the module is imported.

    Returns:
        HuggingFaceEmbeddings: A ready-to-use embedding model.

    Raises:
        EnvironmentError: If HF_TOKEN is missing.
        FileNotFoundError: If the config YAML cannot be found.
        ValueError: If the config YAML is missing the embedding model name.
    """
    global _embed_model_instance

    if _embed_model_instance is not None:
        return _embed_model_instance

    # Validate token before doing anything expensive
    load_hf_token()

    # Locate config
    BASE_DIR = Path(__file__).resolve().parents[2]
    config_path = BASE_DIR / "config" / "model.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}"
        )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_name = config.get("embedding_model", {}).get("model")
    if not model_name:
        raise ValueError(
            "Embedding model name is missing in config/model.yaml "
            "under 'embedding_model.model'."
        )

    use_cuda = config.get("embedding_model", {}).get("cuda", True)
    device = "cuda" if use_cuda else "cpu"

    logger.info("Loading embedding model '%s' on %s ...", model_name, device)

    _embed_model_instance = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model '%s' initialized successfully.", model_name)
    return _embed_model_instance