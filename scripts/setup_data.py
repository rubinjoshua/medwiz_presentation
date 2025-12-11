from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

# Make sure we can import src.* when running as `python scripts/setup_data.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR  # type: ignore  # noqa: E402


def write_sig_examples(path: Path) -> None:
    """Create a JSONL file of synthetic sig examples.

    Each line contains:
      - sig_text: messy or compact free-text sig.
      - english_instructions: clean English.
      - structured_instructions: JSON matching the StructuredSig schema.
    """

    examples = [
        {
            "sig_text": "1 tab po qd x5d",
            "english_instructions": "Take one tablet by mouth once a day for 5 days.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 1,
                        "intake_period": "P1D",
                        "intake_type": "tablet",
                        "duration": "P5D",
                    }
                ]
            },
        },
        {
            "sig_text": "2 tabs po bid x3d",
            "english_instructions": "Take two tablets by mouth twice a day for 3 days.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 2,
                        "intake_period": "P1D",
                        "intake_type": "tablet",
                        "duration": "P3D",
                    }
                ]
            },
        },
        {
            "sig_text": "2x tabs/3d then 1x/2d",
            "english_instructions": "Take two tablets once a day for three days, then one tablet once a day for two days.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 2,
                        "intake_period": "P1D",
                        "intake_type": "tablet",
                        "duration": "P3D",
                    },
                    {
                        "intakes": 1,
                        "intake_period": "P1D",
                        "intake_type": "tablet",
                        "duration": "P2D",
                    },
                ]
            },
        },
        {
            "sig_text": "1 cap po qhs x7d",
            "english_instructions": "Take one capsule by mouth every night at bedtime for 7 days.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 1,
                        "intake_period": "P1D",
                        "intake_type": "capsule",
                        "duration": "P7D",
                    }
                ]
            },
        },
        {
            "sig_text": "5 ml po q8h x3d",
            "english_instructions": "Take 5 milliliters by mouth every 8 hours for 3 days.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 1,
                        "intake_period": "PT8H",
                        "intake_type": "milliliter",
                        "duration": "P3D",
                    }
                ]
            },
        },
        {
            "sig_text": "1 tab po q4h prn pain (max 6/day)",
            "english_instructions": "Take one tablet by mouth every 4 hours as needed for pain, up to 6 tablets per day.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 1,
                        "intake_period": "PT4H",
                        "intake_type": "tablet",
                        "duration": "P1D",
                    }
                ]
            },
        },
        {
            "sig_text": "1 tab po tid x10d",
            "english_instructions": "Take one tablet by mouth three times a day for 10 days.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 3,
                        "intake_period": "P1D",
                        "intake_type": "tablet",
                        "duration": "P10D",
                    }
                ]
            },
        },
        {
            "sig_text": "1 tab po qd x30d",
            "english_instructions": "Take one tablet by mouth once a day for 30 days.",
            "structured_instructions": {
                "sigs": [
                    {
                        "intakes": 1,
                        "intake_period": "P1D",
                        "intake_type": "tablet",
                        "duration": "P30D",
                    }
                ]
            },
        },
    ]

    # Add some simple programmatic variants for more coverage.
    base_patterns = [
        ("tablet", "tab"),
        ("capsule", "cap"),
    ]
    durations = [3, 5, 7]
    daily_intakes = [1, 2]

    for duration in durations:
        for intakes in daily_intakes:
            for full, short in base_patterns:
                sig = f"{intakes} {short} po qd x{duration}d"
                english = (
                    f"Take {intakes} {full}{'s' if intakes > 1 else ''} by mouth once a day "
                    f"for {duration} days."
                )
                examples.append(
                    {
                        "sig_text": sig,
                        "english_instructions": english,
                        "structured_instructions": {
                            "sigs": [
                                {
                                    "intakes": intakes,
                                    "intake_period": "P1D",
                                    "intake_type": full,
                                    "duration": f"P{duration}D",
                                }
                            ]
                        },
                    }
                )

    with path.open("w", encoding="utf-8") as f:
        for record in examples:
            f.write(json.dumps(record) + "\n")


def write_medical_knowledge(path: Path) -> None:
    """Create a small synthetic medical knowledge base CSV.

    Columns: drug_name, form, max_daily_intakes, min_interval_hours, notes
    """

    rows = [
        {
            "drug_name": "Paracetamol 500mg",
            "form": "tablet",
            "max_daily_intakes": 4,
            "min_interval_hours": 4,
            "notes": "Do not exceed 4 doses (2g) per day. Typical adult dose is 1 tablet every 4-6 hours.",
        },
        {
            "drug_name": "Ibuprofen 200mg",
            "form": "tablet",
            "max_daily_intakes": 3,
            "min_interval_hours": 6,
            "notes": "Typical adult dosing is up to 3 doses per day with at least 6 hours between doses.",
        },
        {
            "drug_name": "Amoxicillin 500mg",
            "form": "capsule",
            "max_daily_intakes": 3,
            "min_interval_hours": 8,
            "notes": "Common regimen is one capsule three times daily (every 8 hours).",
        },
        {
            "drug_name": "Atorvastatin 20mg",
            "form": "tablet",
            "max_daily_intakes": 1,
            "min_interval_hours": 24,
            "notes": "Typically taken once daily.",
        },
        {
            "drug_name": "Metformin 500mg",
            "form": "tablet",
            "max_daily_intakes": 3,
            "min_interval_hours": 8,
            "notes": "Often taken once or twice daily with meals, sometimes three times daily.",
        },
        {
            "drug_name": "Omeprazole 20mg",
            "form": "capsule",
            "max_daily_intakes": 1,
            "min_interval_hours": 24,
            "notes": "Typically taken once daily before a meal.",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def write_input_sigs(path: Path) -> None:
    """Create a 10-row demo CSV of semi-structured inputs.

    Columns: patient_name, drug_name, drug_code, sig_text.
    Includes 6 reasonable ("good") sigs and 4 intentionally odd/unsafe ones.
    """

    rows = [
        # Good examples
        {
            "patient_name": "Alice Smith",
            "drug_name": "Paracetamol 500mg",
            "drug_code": "PARA500",
            "sig_text": "1 tab po q6h prn pain (max 4/day)",
        },
        {
            "patient_name": "Bob Jones",
            "drug_name": "Ibuprofen 200mg",
            "drug_code": "IBU200",
            "sig_text": "1 tab po tid with food",
        },
        {
            "patient_name": "Carlos Diaz",
            "drug_name": "Amoxicillin 500mg",
            "drug_code": "AMOX500",
            "sig_text": "1 cap po tid x7d",
        },
        {
            "patient_name": "Dana Lee",
            "drug_name": "Atorvastatin 20mg",
            "drug_code": "ATOR20",
            "sig_text": "1 tab po qhs x30d",
        },
        {
            "patient_name": "Ethan Chen",
            "drug_name": "Metformin 500mg",
            "drug_code": "METF500",
            "sig_text": "1 tab po bid with meals",
        },
        {
            "patient_name": "Fatima Khan",
            "drug_name": "Omeprazole 20mg",
            "drug_code": "OMEP20",
            "sig_text": "1 cap po qam before breakfast x14d",
        },
        # Intentionally problematic / odd examples
        {
            "patient_name": "George Hill",
            "drug_name": "Paracetamol 500mg",
            "drug_code": "PARA500",
            "sig_text": "2 tabs po q2h x5d",
        },
        {
            "patient_name": "Hannah Fox",
            "drug_name": "Ibuprofen 200mg",
            "drug_code": "IBU200",
            "sig_text": "3 tabs po q4h x3d",
        },
        {
            "patient_name": "Ian Wright",
            "drug_name": "Amoxicillin 500mg",
            "drug_code": "AMOX500",
            "sig_text": "1 cap po q2h x10d",
        },
        {
            "patient_name": "Julia Park",
            "drug_name": "Atorvastatin 20mg",
            "drug_code": "ATOR20",
            "sig_text": "1 tab po tid x90d",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sig_examples_path = DATA_DIR / "sig_examples.jsonl"
    med_kb_path = DATA_DIR / "medical_knowledge.csv"
    input_sigs_path = DATA_DIR / "input_sigs.csv"

    print(f"Writing sig examples to {sig_examples_path}")
    write_sig_examples(sig_examples_path)

    print(f"Writing medical knowledge base to {med_kb_path}")
    write_medical_knowledge(med_kb_path)

    print(f"Writing demo input sigs to {input_sigs_path}")
    write_input_sigs(input_sigs_path)

    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
