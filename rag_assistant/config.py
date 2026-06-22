import os
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
# ─── Cesty ───────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
NOTEBOOKS_DIR   = DATA_DIR / "notebooks"

# ─── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#OLLAMA_BASE_URL = "http://host.docker.internal:11434"
LLM_MODEL        = "gemma3:4b"

# Multilinguálny embedding model (slovenčina + angličtina)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL   = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ─── RAG parametre ────────────────────────────────────────────────────────────
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
TOP_K_RETRIEVAL  = 5        # Počet relevantných chunks
MEMORY_K         = 3        # Počet správ v pamäti (dozadu)
SIMILARITY_THRESHOLD = 0.35  # Minimálna podobnosť pre zdroj

# ─── Podporované formáty ──────────────────────────────────────────────────────
SUPPORTED_FORMATS = [".pdf", ".xlsx", ".xls", ".docx"]

# Vytvorenie priečinkov
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def _detect_gpu() -> bool:
    """Zistí, či je dostupná NVIDIA GPU."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_total_ram_gb() -> float:
    """Vráti celkovú RAM v GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 16.0  # Bezpečný default — predpokladáme dostatok pamäte


def _autodetect_llm_model() -> tuple[str, bool]:
    """
    Vyberie LLM model podľa dostupného hardvéru.
    
    Returns:
        (model_name, is_reduced_quality)
        - is_reduced_quality=True znamená, že sa použil úsporný model
          a aplikácia by mala upozorniť používateľa.
    """
    # Manuálny override má prioritu
    if env_model := os.getenv("LLM_MODEL"):
        logger.info(f"LLM model nastavený manuálne: {env_model}")
        return env_model, False

    has_gpu = _detect_gpu()
    total_ram_gb = _get_total_ram_gb()

    logger.info(f"Hardvér: GPU={has_gpu}, RAM={total_ram_gb:.1f} GB")

    # Slabý hardvér → menší model s upozornením
    if total_ram_gb < 12:
        logger.warning(
            f"Slabý hardvér detegovaný (no GPU, {total_ram_gb:.1f} GB RAM). "
            f"Použije sa úsporný model gemma3:1b s nižšou kvalitou odpovedí."
        )
        return "gemma3:1b", True

    # Štandardný prípad
    return "gemma3:4b", False


LLM_MODEL, IS_REDUCED_QUALITY = _autodetect_llm_model()


def get_hardware_info() -> dict:
    """Pre zobrazenie v UI alebo logoch."""
    return {
        "total_ram_gb": round(_get_total_ram_gb(), 1),
        "has_gpu": _detect_gpu(),
        "selected_model": LLM_MODEL,
        "reduced_quality": IS_REDUCED_QUALITY,
    }
