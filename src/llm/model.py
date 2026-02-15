from langchain_ollama import OllamaLLM
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

# Lazy-loaded singleton for the Ollama LLM

_llm_instance = None


def get_llm():
    """
    Return a lazily-initialized singleton of the Ollama LLM.

    The model client is created on the first call and reused for all
    subsequent calls, avoiding redundant initialisation on every import.

    Returns:
        OllamaLLM: A LangChain-compatible LLM instance.

    Raises:
        FileNotFoundError: If the config YAML cannot be found.
        ValueError: If required LLM settings are missing in the config.
        ConnectionError: If the Ollama server is unreachable.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    # Locate config
    BASE_DIR = Path(__file__).resolve().parents[2]
    config_path = BASE_DIR / "config" / "model.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}"
        )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    llm_cfg = config.get("llm_config", {})

    model_name = llm_cfg.get("model")
    base_url = llm_cfg.get("base_url")

    if not model_name:
        raise ValueError(
            "LLM model name is missing in config/model.yaml "
            "under 'llm_config.model'."
        )
    if not base_url:
        raise ValueError(
            "LLM base_url is missing in config/model.yaml "
            "under 'llm_config.base_url'."
        )

    logger.info(
        "Initializing Ollama LLM '%s' at %s ...", model_name, base_url
    )

    _llm_instance = OllamaLLM(
        model=model_name,
        base_url=base_url,
        temperature=llm_cfg.get("temperature", 0.5),
        num_ctx=llm_cfg.get("num_ctx", 4096),
        num_gpu=llm_cfg.get("num_gpu", 1),
    )

    logger.info(
        "Ollama LLM '%s' initialized (temp=%s, base_url=%s).",
        _llm_instance.model,
        _llm_instance.temperature,
        _llm_instance.base_url,
    )
    return _llm_instance