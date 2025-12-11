from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOllama:
    """Return a cached ChatOllama instance.

    Expects the Ollama daemon to be running locally and the model to be pulled
    beforehand, e.g. `ollama pull llama3.2`.
    """

    return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.1)


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    """Return a cached OllamaEmbeddings instance.

    Uses the same underlying model as the chat LLM for simplicity.
    """

    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
