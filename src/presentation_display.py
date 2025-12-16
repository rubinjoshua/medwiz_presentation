from __future__ import annotations

import threading
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table

from .config import DATA_DIR, DISPLAY_REFRESH_SECONDS

console = Console()


def _make_translation_table(df: pd.DataFrame) -> Table:
    table = Table(title="AI translations (pre-validation)", expand=True)

    # Keep the rows/columns/content consistent with the translation-only view.
    table.add_column("sig", style="cyan", overflow="fold", ratio=1)
    table.add_column("english_translation", style="green", overflow="fold", ratio=2)
    table.add_column("structured_sig", style="white", overflow="fold", ratio=3)

    for row in df.itertuples(index=False):
        table.add_row(
            str(getattr(row, "sig", "")),
            str(getattr(row, "english_translation", "")),
            str(getattr(row, "structured_sig", "")),
        )

    return table


def _make_output_table(df: pd.DataFrame) -> Table:
    table = Table(title="AI translated + validated prescriptions")

    # Column order:
    # - ai_validation: emoji
    # - validation_decision: brief explanation text (from validation_reason)
    columns = [
        ("patient_name", "white"),
        ("drug_name", "white"),
        ("drug_code", "white"),
        ("sig_text", "cyan"),
        ("english_instructions", "green"),
        ("ai_validation", "bold"),
        ("validation_decision", "magenta"),
    ]

    for name, style in columns:
        table.add_column(name, style=style, overflow="fold")

    for row in df.itertuples(index=False):
        ai_validation = getattr(row, "ai_validation", "") or getattr(row, "ai_validated_emoji", "")
        validation_text = getattr(row, "validation_reason", "") or getattr(row, "validation_decision", "")

        table.add_row(
            str(getattr(row, "patient_name", "")),
            str(getattr(row, "drug_name", "")),
            str(getattr(row, "drug_code", "")),
            str(getattr(row, "sig_text", "")),
            str(getattr(row, "english_instructions", "")),
            str(ai_validation),
            str(validation_text),
        )
    return table


def _build_layout(translated_df: pd.DataFrame, output_df: pd.DataFrame) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=1),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].update(_make_translation_table(translated_df))
    layout["bottom"].update(_make_output_table(output_df))
    return layout


def _display_loop(translated_path: Path, output_path: Path, stop_event: threading.Event) -> None:
    """Background loop that keeps the tables refreshed."""

    with Live(console=console, screen=True, auto_refresh=False) as live:
        while not stop_event.is_set():
            if translated_path.exists():
                translated_df = pd.read_csv(translated_path)
            else:
                translated_df = pd.DataFrame(columns=["sig", "english_translation", "structured_sig"])

            if output_path.exists():
                output_df = pd.read_csv(output_path)
            else:
                output_df = pd.DataFrame(
                    columns=[
                        "patient_name",
                        "drug_name",
                        "drug_code",
                        "sig_text",
                        "english_instructions",
                        "ai_validation",
                        "validation_decision",
                        "validation_reason",
                        # Back-compat (older output files)
                        "ai_validated_emoji",
                    ]
                )

            layout = _build_layout(translated_df, output_df)
            live.update(layout, refresh=True)
            time.sleep(DISPLAY_REFRESH_SECONDS)




def run_display() -> None:
    """Continuously display the translation + validated output tables.

    Intended to run in the right-hand pane during the presentation.
    """

    translated_path = DATA_DIR / "translated_sigs.csv"
    output_path = DATA_DIR / "output_sigs.csv"

    stop_event = threading.Event()
    display_thread = threading.Thread(
        target=_display_loop, args=(translated_path, output_path, stop_event), daemon=True
    )
    display_thread.start()

    try:
        # Keep the main thread alive so Ctrl+C works.
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping live display...[/yellow]")
        stop_event.set()
        display_thread.join(timeout=1.0)
