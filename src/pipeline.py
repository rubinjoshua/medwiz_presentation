from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, SLEEP_TRANSLATION_STAGE, SLEEP_VALIDATION_STAGE
from .sig_translation import translate_sig
from .sig_validation import validate_sig


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _pretty_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def run_demo() -> None:
    """Run the full demo pipeline over the 10-row input_sigs.csv.

    For each row, this will:
    - Show the raw sig.
    - Show retrieved similar examples used for translation.
    - Show the LLM translation result (English + structured JSON).
    - Show retrieved medical knowledge for validation.
    - Show the validation decision and reason.
    - Append the result to data/output_sigs.csv so the display process can update.
    """

    input_path = DATA_DIR / "input_sigs.csv"
    output_path = DATA_DIR / "output_sigs.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected input file {input_path} to exist. Run `python scripts/setup_data.py` first."
        )

    df = pd.read_csv(input_path)

    results_rows = []

    _print_header("Starting Medwiz sig translation + validation demo")
    print(f"Reading {len(df)} prescriptions from {input_path}...\n")

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        patient_name = str(row.patient_name)
        drug_name = str(row.drug_name)
        drug_code = str(row.drug_code)
        sig_text = str(row.sig_text)

        _print_header(f"Prescription {idx}/{len(df)} — {patient_name} — {drug_name} ({drug_code})")
        print(f"Raw sig text: {sig_text}\n")
        time.sleep(SLEEP_TRANSLATION_STAGE)

        # --- Translation stage ---
        print("[Translation] Retrieving similar sig examples from local vector DB...\n")
        translation_result, example_docs = translate_sig(sig_text)

        for i, doc in enumerate(example_docs, start=1):
            meta = doc.metadata or {}
            print(f"Example {i} sig: {doc.page_content}")
            if meta.get("english_instructions"):
                print(f"  English: {meta['english_instructions']}")
            if meta.get("structured_instructions") is not None:
                print("  Structured:")
                print(_pretty_json(meta["structured_instructions"]))
            print()
        time.sleep(SLEEP_TRANSLATION_STAGE)

        print("[Translation] Sending to local LLM for structured translation... (via Ollama)\n")
        time.sleep(SLEEP_TRANSLATION_STAGE)

        print("[Translation] LLM produced this English explanation:")
        print(f"  {translation_result.english_instructions}\n")
        print("[Translation] And this structured JSON representation:")
        print(_pretty_json(translation_result.structured.model_dump()) + "\n")
        time.sleep(SLEEP_TRANSLATION_STAGE)

        # --- Validation stage ---
        print("[Validation] Retrieving relevant medical knowledge from local vector DB...\n")
        validation_result, ref_docs = validate_sig(drug_name=drug_name, english_instructions=translation_result.english_instructions)

        for i, doc in enumerate(ref_docs, start=1):
            print(f"Reference {i}: {doc.page_content}\n")
        time.sleep(SLEEP_VALIDATION_STAGE)

        print("[Validation] Asking local LLM to judge if this dosing is acceptable...\n")
        time.sleep(SLEEP_VALIDATION_STAGE)

        print("[Validation] Decision:")
        print(f"  Verdict: {validation_result.decision} {validation_result.emoji}")
        print(f"  Reason: {validation_result.reason}\n")

        # --- Collect and persist output row ---
        output_row = {
            "patient_name": patient_name,
            "drug_name": drug_name,
            "drug_code": drug_code,
            "sig_text": sig_text,
            "english_instructions": translation_result.english_instructions,
            "structured_instructions_json": json.dumps(
                translation_result.structured.model_dump(), ensure_ascii=False
            ),
            "validation_decision": validation_result.decision,
            "validation_reason": validation_result.reason,
            "ai_validated_emoji": validation_result.emoji,
        }
        results_rows.append(output_row)

        output_df = pd.DataFrame(results_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)

        print(f"[Output] Appended result to {output_path} and saved.")
        print("[Output] The right-hand table view should now show this row.\n")
        time.sleep(1.0)

    _print_header("Demo complete")
    print(f"Processed {len(df)} prescriptions. Final results written to {output_path}.")
