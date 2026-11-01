"""
config.py — Centralized configuration for FactSense Phase 3
-----------------------------------------------------------
Unifies model names, environment variables, and active provider tracking.
"""

import os
import logging
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Default Model Names
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
DEFAULT_NLI_MODEL = os.environ.get("FACTSENSE_NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")

# Feature Flags
USE_DEBERTA_CROSSCHECK = os.environ.get("USE_DEBERTA_CROSSCHECK", "false").strip().lower() in ("true", "1", "yes")

# Relevance & Confidence Thresholds
RAW_SIMILARITY_THRESHOLD = float(os.environ.get("RAW_SIMILARITY_THRESHOLD", "0.35"))
MAX_CLAIMS_DEFAULT = 6
PASSAGE_CHAR_WINDOW = 900


def get_llm_config() -> Dict[str, Any]:
    """Detects which LLM provider is active."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    ollama_model = os.environ.get("OLLAMA_MODEL")

    if gemini_key:
        return {
            "provider": "gemini",
            "model": DEFAULT_GEMINI_MODEL,
            "api_key": gemini_key,
            "degraded": False
        }
    elif groq_key:
        return {
            "provider": "groq",
            "model": DEFAULT_GROQ_MODEL,
            "api_key": groq_key,
            "degraded": False
        }
    elif openai_key:
        return {
            "provider": "openai",
            "model": DEFAULT_OPENAI_MODEL,
            "api_key": openai_key,
            "degraded": False
        }
    elif ollama_model:
        return {
            "provider": "ollama",
            "model": ollama_model,
            "api_key": None,
            "degraded": False
        }
    else:
        return {
            "provider": None,
            "model": None,
            "api_key": None,
            "degraded": True
        }


def get_active_providers() -> Dict[str, Any]:
    """Returns a status dictionary of all active components."""
    llm_info = get_llm_config()
    
    search_providers = []
    if os.environ.get("SERPER_API_KEY"):
        search_providers.append("serper")
    if os.environ.get("TAVILY_API_KEY"):
        search_providers.append("tavily")
    search_providers.extend(["duckduckgo", "wikipedia"])

    return {
        "llm_configured": not llm_info["degraded"],
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "search_providers": search_providers,
        "use_deberta_crosscheck": USE_DEBERTA_CROSSCHECK,
        "degraded": llm_info["degraded"]
    }


def extract_text_from_response(response: Any) -> str:
    """Helper to safely extract raw text from LangChain message or response object."""
    if response is None:
        return ""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Format: [{'type': 'text', 'text': '...'}]
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts).strip()
    return str(content).strip()


def get_llm(temperature: float = 0.0) -> Optional[Any]:
    """Instantiates the configured LLM client or returns None if in degraded mode."""
    cfg = get_llm_config()
    provider = cfg["provider"]

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=cfg["model"],
                google_api_key=cfg["api_key"],
                temperature=temperature,
                request_timeout=15.0,
                max_retries=1
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            return None

    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=cfg["model"],
                groq_api_key=cfg["api_key"],
                temperature=temperature,
                request_timeout=12.0
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            return None

    elif provider == "openai":
        try:
            from langchain_community.chat_models import ChatOpenAI
            return ChatOpenAI(
                model_name=cfg["model"],
                openai_api_key=cfg["api_key"],
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            return None

    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return ChatOllama(
                model=cfg["model"],
                base_url=base_url,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            return None

    return None

