from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .config import (
    DATA_DIR,
    VECTOR_DB_DIR,
    SIG_COLLECTION_NAME,
    MED_KB_COLLECTION_NAME,
    SIG_K,
    MED_K,
)
from .llm import get_embedding_model


SIG_DB_DIR = VECTOR_DB_DIR / "sig_examples"
MED_DB_DIR = VECTOR_DB_DIR / "medical_knowledge"


def _ensure_dir(path: Path, rebuild: bool) -> None:
    if rebuild and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_sig_examples_store(rebuild: bool = True) -> Chroma:
    """Build and persist the Chroma store for sig examples.

    Expects `data/sig_examples.jsonl` to exist.
    """

    _ensure_dir(SIG_DB_DIR, rebuild=rebuild)

    sig_examples_path = DATA_DIR / "sig_examples.jsonl"
    if not sig_examples_path.exists():
        raise FileNotFoundError(f"Missing sig examples file: {sig_examples_path}")

    docs: list[Document] = []
    with sig_examples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs.append(
                Document(
                    page_content=record["sig_text"],
                    metadata={
                        "english_instructions": record["english_instructions"],
                        "structured_instructions": record["structured_instructions"],
                    },
                )
            )

    embeddings = get_embedding_model()
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        collection_name=SIG_COLLECTION_NAME,
        persist_directory=str(SIG_DB_DIR),
    )
    vectordb.persist()
    return vectordb


def build_med_knowledge_store(rebuild: bool = True) -> Chroma:
    """Build and persist the Chroma store for the medical knowledge base."""

    _ensure_dir(MED_DB_DIR, rebuild=rebuild)

    med_kb_path = DATA_DIR / "medical_knowledge.csv"
    if not med_kb_path.exists():
        raise FileNotFoundError(f"Missing medical knowledge file: {med_kb_path}")

    df = pd.read_csv(med_kb_path)

    docs: list[Document] = []
    for row in df.itertuples(index=False):
        snippet = (
            f"Drug: {row.drug_name} ({row.form}). "
            f"Max daily intakes: {row.max_daily_intakes}. "
            f"Minimum interval: {row.min_interval_hours} hours. "
            f"Notes: {row.notes}"
        )
        docs.append(
            Document(
                page_content=snippet,
                metadata={
                    "drug_name": row.drug_name,
                    "form": row.form,
                    "max_daily_intakes": int(row.max_daily_intakes),
                    "min_interval_hours": float(row.min_interval_hours),
                    "notes": row.notes,
                },
            )
        )

    embeddings = get_embedding_model()
    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        collection_name=MED_KB_COLLECTION_NAME,
        persist_directory=str(MED_DB_DIR),
    )
    vectordb.persist()
    return vectordb


def get_sig_examples_vectorstore() -> Chroma:
    """Load the persisted sig examples store."""

    embeddings = get_embedding_model()
    return Chroma(
        collection_name=SIG_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(SIG_DB_DIR),
    )


def get_med_knowledge_vectorstore() -> Chroma:
    """Load the persisted medical knowledge store."""

    embeddings = get_embedding_model()
    return Chroma(
        collection_name=MED_KB_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(MED_DB_DIR),
    )


def get_sig_examples_retriever():
    return get_sig_examples_vectorstore().as_retriever(search_kwargs={"k": SIG_K})


def get_med_knowledge_retriever():
    return get_med_knowledge_vectorstore().as_retriever(search_kwargs={"k": MED_K})
