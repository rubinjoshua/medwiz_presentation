from __future__ import annotations

import json
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .llm import get_chat_llm
from .models import TranslationResult, StructuredSig
from .vectorstores import get_sig_examples_retriever


_llm = get_chat_llm()

# NOTE: We intentionally do NOT use PydanticOutputParser.get_format_instructions() here.
# For small local models (e.g. llama3.2), the generated JSON schema is large and the
# model often "forgets" to emit english_instructions and returns only the inner
# {"sigs": [...]} object. We instead provide a short, explicit contract.
_TRANSLATION_FORMAT_INSTRUCTIONS = (
    "Return a single JSON object with EXACTLY these top-level keys:\n"
    "- english_instructions (string)\n"
    "- structured (object)\n\n"
    "The structured object MUST be of the form:\n"
    "{\"sigs\": [{\"intakes\": <int>, \"intake_period\": <ISO-8601 duration string>, "
    "\"intake_type\": <string>, \"duration\": <ISO-8601 duration string>}]}\n\n"
    "Do not wrap in markdown. Do not return only the structured object."
)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert clinical pharmacist. You translate compact, messy "
            "prescription instructions (sigs) into clear English and a structured JSON format.\n\n"
            "Return ONLY JSON.\n\n"
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

# Secondary prompt used only as a fallback when the model returns structured JSON
# without english_instructions.
_english_only_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert clinical pharmacist. Translate the raw sig into a clear, "
            "patient-friendly English instruction. Return ONLY JSON of the form: "
            "{\"english_instructions\": \"...\"}.",
        ),
        (
            "human",
            "Here are similar past examples:\n\n{examples_block}\n\n"
            "Raw sig: {sig_text}",
        ),
    ]
)


def _build_examples_block(docs: List[Document]) -> str:
    """Build an examples block.

    IMPORTANT: Examples are shown as full JSON outputs (not separate English/Structured
    lines) because many small models will otherwise "learn" that only the last JSON-ish
    thing matters and return just {"sigs": [...] }.
    """

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        english = str(meta.get("english_instructions", "")).strip()
        structured = meta.get("structured_instructions", "")

        # In the vector store we keep structured_instructions as a JSON string
        # to satisfy Chroma's metadata constraints.
        structured_obj: object
        if isinstance(structured, str):
            try:
                structured_obj = json.loads(structured)
            except json.JSONDecodeError:
                structured_obj = structured
        else:
            structured_obj = structured

        example_out = {"english_instructions": english, "structured": structured_obj}
        parts.append(
            f"Example {i}:\n"
            f"Input sig: {doc.page_content}\n"
            f"Output JSON: {json.dumps(example_out, ensure_ascii=False)}"
        )

    return "\n\n".join(parts) if parts else "(No prior examples available.)"


def _parse_translation_output(raw: str) -> TranslationResult:
    """Parse the LLM's raw JSON (or JSON-in-markdown) into TranslationResult.

    We try to be forgiving because small local models sometimes ignore parts
    of the formatting instructions.

    Common failure mode: valid JSON followed by extra trailing characters
    (e.g. an extra '}' or a short explanation), which triggers
    json.decoder.JSONDecodeError: Extra data.
    """

    text = raw.strip()

    # Strip Markdown code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    def _loads_first_json_object(s: str) -> object:
        """Parse the first JSON value in a string, ignoring any trailing text."""

        decoder = json.JSONDecoder()

        # First try parsing from the beginning.
        try:
            obj, _end = decoder.raw_decode(s.lstrip())
            return obj
        except json.JSONDecodeError:
            pass

        # Otherwise, try starting from the first '{' (helps if the model
        # prepends text like 'Sure, here is the JSON:').
        start = s.find("{")
        if start != -1:
            obj, _end = decoder.raw_decode(s[start:])
            return obj

        raise json.JSONDecodeError("No JSON object found", s, 0)

    try:
        obj = _loads_first_json_object(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM JSON output: {e}\nRaw: {raw}") from e

    if isinstance(obj, dict) and "english_instructions" in obj and "structured" in obj:
        structured = StructuredSig.model_validate(obj["structured"])
        return TranslationResult(
            english_instructions=str(obj["english_instructions"]),
            structured=structured,
        )

    # Fallback: model only returned the structured part like {"sigs": [...]}.
    # We return an empty english_instructions so the caller can do a targeted
    # second attempt to generate just the English text.
    if isinstance(obj, dict) and "sigs" in obj:
        structured = StructuredSig.model_validate(obj)
        return TranslationResult(english_instructions="", structured=structured)

    raise ValueError(f"Unexpected JSON schema from LLM: {obj}")


def _generate_english_only(sig_text: str, examples_block: str) -> str:
    """Fallback: ask the model for English only.

    This is used when the primary translation call returns only {"sigs": ...}.
    """

    prompt_value = _english_only_prompt.invoke(
        {
            "examples_block": examples_block,
            "sig_text": sig_text,
        }
    )
    response = _llm.invoke(prompt_value)
    raw = getattr(response, "content", response)

    text = str(raw).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # If the model ignored the JSON-only instruction, just return the raw text.
        return text

    if isinstance(obj, dict) and "english_instructions" in obj:
        return str(obj["english_instructions"]).strip()

    return text


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
    # with a .to_messages() method (required in LangChain ≥0.3).
    prompt_value = _prompt.invoke(
        {
            "format_instructions": _TRANSLATION_FORMAT_INSTRUCTIONS,
            "examples_block": examples_block,
            "sig_text": sig_text,
        }
    )
    got_response = False
    while True:
        response = _llm.invoke(prompt_value)
        raw = getattr(response, "content", response)
        try:
            result = _parse_translation_output(str(raw))
            got_response = True
        except ValueError as e:
            print(f"Got exception on parse: \n{e}\n, trying llm call again...")
        if got_response:
            break

    # If the model returned only structured JSON, do a cheap second call to
    # generate the missing English instructions.
    if not (result.english_instructions or "").strip():
        result.english_instructions = _generate_english_only(sig_text=sig_text, examples_block=examples_block)

    return result, docs
