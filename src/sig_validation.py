from __future__ import annotations

from typing import List, Tuple
import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .llm import get_chat_llm
from .models import ValidationResult
from .vectorstores import get_med_knowledge_retriever


_llm = get_chat_llm()
# As with translation, use the parser only for generating formatting
# instructions; we will parse manually for robustness.
_parser = PydanticOutputParser(pydantic_object=ValidationResult)
_format_instructions = _parser.get_format_instructions()

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a cautious clinical pharmacist. You check whether dosing "
            "instructions (sigs) are reasonable for a specific drug, using given "
            "reference information.\n\n"
            "You must answer in JSON ONLY, matching the schema.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Drug name: {drug_name}\n"
            "New case instructions: {english_instructions}\n\n"
            "Here is reference information about this and similar drugs:\n\n{reference_block}\n\n"
            "Based ONLY on the reference information and basic pharmacologic safety, "
            "decide whether the dosing instructions are acceptable. "
            "Set decision to 'OK' if clearly acceptable, otherwise 'NOT_OK'. "
            "Give a short, clear reason.",
        ),
    ]
)


def _build_reference_block(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"Reference {i}: {doc.page_content}")
    return "\n\n".join(parts) if parts else "(No reference information available.)"


def _parse_validation_output(raw: str) -> ValidationResult:
    """Parse LLM output into ValidationResult, with fallbacks.

    We try JSON first; if that fails, we infer decision from the text.
    """

    text = raw.strip()

    # Strip Markdown fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat raw text as the reason, infer decision heuristically.
        upper = text.upper()
        if "NOT_OK" in upper or "NOT OK" in upper or "UNSAFE" in upper or "TOO MUCH" in upper:
            decision = "NOT_OK"
        else:
            decision = "OK"
        return ValidationResult(decision=decision, reason=text, emoji="")

    if isinstance(obj, dict):
        decision = obj.get("decision") or obj.get("verdict") or obj.get("status")
        reason = obj.get("reason") or obj.get("explanation") or ""
        if not decision:
            upper = text.upper()
            if "NOT_OK" in upper or "NOT OK" in upper or "UNSAFE" in upper or "TOO MUCH" in upper:
                decision = "NOT_OK"
            else:
                decision = "OK"
        return ValidationResult(
            decision=str(decision),
            reason=str(reason) if reason else text,
            emoji="",
        )

    # If the model returned something unexpected, fall back to OK with the raw text.
    return ValidationResult(decision="OK", reason=text, emoji="")


def validate_sig(drug_name: str, english_instructions: str) -> Tuple[ValidationResult, List[Document]]:
    """Validate instructions for a given drug using the local medical knowledge base.

    Returns the parsed ValidationResult and the retrieved reference documents
    so the caller can show them in the presentation logs.
    """

    query = f"The instructions: {english_instructions}, were given for this drug {drug_name}."

    retriever = get_med_knowledge_retriever()
    # In LangChain ≥0.3, retrievers are Runnables; use .invoke() to get
    # relevant documents instead of get_relevant_documents().
    docs = retriever.invoke(query)
    reference_block = _build_reference_block(docs)

    prompt_value = _prompt.invoke(
        {
            "format_instructions": _format_instructions,
            "drug_name": drug_name,
            "english_instructions": english_instructions,
            "reference_block": reference_block,
        }
    )
    response = _llm.invoke(prompt_value)
    raw = getattr(response, "content", response)

    result = _parse_validation_output(str(raw))

    # Map decision to emoji here for convenience (LLM can ignore emoji logic).
    decision_upper = (result.decision or "").strip().upper()
    if decision_upper == "OK":
        emoji = "✅"
    else:
        emoji = "❌"

    result.emoji = emoji
    result.decision = "OK" if decision_upper == "OK" else "NOT_OK"
    return result, docs
