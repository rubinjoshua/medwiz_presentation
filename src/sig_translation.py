from __future__ import annotations

import json
from typing import List, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .llm import get_chat_llm
from .models import TranslationResult
from .vectorstores import get_sig_examples_retriever


_llm = get_chat_llm()
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

_chain = _prompt | _llm | _parser


def _build_examples_block(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        english = meta.get("english_instructions", "")
        structured = meta.get("structured_instructions", {})
        structured_str = json.dumps(structured, ensure_ascii=False)
        parts.append(
            f"Example {i}:\n"
            f"Raw sig: {doc.page_content}\n"
            f"English: {english}\n"
            f"Structured: {structured_str}"
        )
    return "\n\n".join(parts) if parts else "(No prior examples available.)"


def translate_sig(sig_text: str) -> Tuple[TranslationResult, List[Document]]:
    """Translate a raw sig string into English + structured JSON.

    Returns the parsed TranslationResult and the retrieved example documents
    (so the caller can show them in the presentation logs).
    """

    retriever = get_sig_examples_retriever()
    docs = retriever.get_relevant_documents(sig_text)
    examples_block = _build_examples_block(docs)

    result: TranslationResult = _chain.invoke(
        {"format_instructions": _format_instructions, "examples_block": examples_block, "sig_text": sig_text}
    )
    return result, docs
