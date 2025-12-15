from __future__ import annotations

import json
from typing import List, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .llm import get_chat_llm
from .models import TranslationResult, StructuredSig
from .vectorstores import get_sig_examples_retriever


_llm = get_chat_llm()
# We only use the parser for nice formatting instructions; we will parse
# the JSON manually to be robust to small local models.
_parser = PydanticOutputParser(pydantic_object=TranslationResult)
_format_instructions = _parser.get_format_instructions()

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert clinical pharmacist. You translate compact, messy "
            "prescription instructions (sigs) into clear English and a structured JSON format.\n\n"
            "Return ONLY JSON that matches the specified schema. Do not include any extra text.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Here are similar past examples to guide you:\n\n{examples_block}\n\n"
            "Now translate this new sig.\n\n"
            "Raw sig: {sig_text}",
        ),
    ]
)


def _build_examples_block(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        english = meta.get("english_instructions", "")
        structured = meta.get("structured_instructions", "")
        # In the vector store we keep structured_instructions as a JSON string
        # to satisfy Chroma's metadata constraints. If for some reason we get
        # a dict here, fall back to dumping it.
        if isinstance(structured, str):
            structured_str = structured
        else:
            structured_str = json.dumps(structured, ensure_ascii=False)
        parts.append(
            f"Example {i}:\n"
            f"Raw sig: {doc.page_content}\n"
            f"English: {english}\n"
            f"Structured: {structured_str}"
        )
    return "\n\n".join(parts) if parts else "(No prior examples available.)"


def _parse_translation_output(raw: str, sig_text: str) -> TranslationResult:
    """Parse the LLM's raw JSON (or JSON-in-markdown) into TranslationResult.

    We try to be forgiving because small local models sometimes ignore parts
    of the formatting instructions.
    """

    text = raw.strip()

    # Strip Markdown code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    # Try to parse JSON; if that fails, try to salvage the first {...} block.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        if "{" in text and "}" in text:
            candidate = text[text.find("{") : text.rfind("}") + 1]
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError as e:  # pragma: no cover - defensive
                raise ValueError(f"Could not parse LLM JSON output: {e}\nRaw: {raw}") from e
        else:  # pragma: no cover - very unlikely for this demo
            raise ValueError(f"LLM did not return JSON. Raw output: {raw}")

    if isinstance(obj, dict) and "english_instructions" in obj and "structured" in obj:
        structured = StructuredSig.model_validate(obj["structured"])
        return TranslationResult(
            english_instructions=str(obj["english_instructions"]),
            structured=structured,
        )

    # Fallback: model only returned the structured part like {"sigs": [...]}.
    if isinstance(obj, dict) and "sigs" in obj:
        structured = StructuredSig.model_validate(obj)
        # Use the raw sig_text as the "English" fallback for this demo.
        return TranslationResult(english_instructions=sig_text, structured=structured)

    raise ValueError(f"Unexpected JSON schema from LLM: {obj}")


def translate_sig(sig_text: str) -> Tuple[TranslationResult, List[Document]]:
    """Translate a raw sig string into English + structured JSON.

    Returns the parsed TranslationResult and the retrieved example documents
    (so the caller can show them in the presentation logs).
    """

    retriever = get_sig_examples_retriever()
    # In LangChain ≥0.3, retrievers are Runnables; use .invoke() to get
    # relevant documents instead of get_relevant_documents().
    docs = retriever.invoke(sig_text)
    examples_block = _build_examples_block(docs)

    # Build a ChatPromptValue so the chat model receives a message object
    # with a .to_messages() method (required in LangChain 0.3).
    prompt_value = _prompt.invoke(
        {
            "format_instructions": _format_instructions,
            "examples_block": examples_block,
            "sig_text": sig_text,
        }
    )
    response = _llm.invoke(prompt_value)
    raw = getattr(response, "content", response)

    result = _parse_translation_output(str(raw), sig_text=sig_text)
    return result, docs
