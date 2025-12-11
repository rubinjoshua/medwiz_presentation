from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table

from .config import DATA_DIR, DISPLAY_REFRESH_SECONDS

console = Console()


def _make_input_table(df: pd.DataFrame) -> Table:
    table = Table(title="Incoming prescriptions (semi-structured)")
    for col in ["patient_name", "drug_name", "drug_code", "sig_text"]:
        if col in df.columns:
            table.add_column(col, style="cyan", overflow="fold")

    for row in df.itertuples(index=False):
        table.add_row(
            str(getattr(row, "patient_name", "")),
            str(getattr(row, "drug_name", "")),
            str(getattr(row, "drug_code", "")),
            str(getattr(row, "sig_text", "")),
        )
    return table


def _make_output_table(df: pd.DataFrame) -> Table:
    table = Table(title="AI translated + validated prescriptions")

    columns = [
        ("patient_name", "white"),
        ("drug_name", "white"),
        ("drug_code", "white"),
        ("sig_text", "cyan"),
        ("english_instructions", "green"),
        ("validation_decision", "magenta"),
        ("ai_validated_emoji", "bold"),
    ]

    for name, style in columns:
        if name in df.columns:
            table.add_column(name, style=style, overflow="fold")

    for row in df.itertuples(index=False):
        table.add_row(
            str(getattr(row, "patient_name", "")),
            str(getattr(row, "drug_name", "")),
            str(getattr(row, "drug_code", "")),
            str(getattr(row, "sig_text", "")),
            str(getattr(row, "english_instructions", "")),
            str(getattr(row, "validation_decision", "")),
            str(getattr(row, "ai_validated_emoji", "")),
        )
    return table


def _build_layout(input_df: pd.DataFrame, output_df: pd.DataFrame) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=1),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].update(_make_input_table(input_df))
    layout["bottom"].update(_make_output_table(output_df))
    return layout


def run_display() -> None:
    """Continuously display the input and output tables.

    Intended to run in the right-hand pane during the presentation.
    """

    input_path = DATA_DIR / "input_sigs.csv"
    output_path = DATA_DIR / "output_sigs.csv"

    if not input_path.exists():
        console.print(f"[red]Expected input file {input_path} to exist. Run `python scripts/setup_data.py` first.[/red]")
        return

    input_df = pd.read_csv(input_path)

    console.print("Starting live display. Waiting for output_sigs.csv to be created...\n")

    with Live(console=console, screen=True, auto_refresh=False) as live:
        while True:
            if output_path.exists():
                output_df = pd.read_csv(output_path)
            else:
                output_df = pd.DataFrame(columns=[
                    "patient_name",
                    "drug_name",
                    "drug_code",
                    "sig_text",
                    "english_instructions",
                    "validation_decision",
                    "ai_validated_emoji",
                ])

            layout = _build_layout(input_df, output_df)
            live.update(layout, refresh=True)
            time.sleep(DISPLAY_REFRESH_SECONDS)
