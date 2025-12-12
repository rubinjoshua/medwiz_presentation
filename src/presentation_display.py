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
        Layout(name="middle", ratio=1),
        Layout(name="bottom", size=5),
    )
    layout["top"].update(_make_input_table(input_df))
    layout["middle"].update(_make_output_table(output_df))

    help_table = Table(show_header=False, box=None)
    help_table.add_column("msg", style="yellow")
    help_table.add_row(
        "Type new prescriptions in the terminal below while this screen is running. "
        "They will appear in the top table as soon as you add them."
    )
    layout["bottom"].update(help_table)
    return layout


def _display_loop(input_path: Path, output_path: Path, stop_event: threading.Event) -> None:
    """Background loop that keeps the tables refreshed."""

    with Live(console=console, screen=True, auto_refresh=False) as live:
        while not stop_event.is_set():
            if input_path.exists():
                input_df = pd.read_csv(input_path)
            else:
                input_df = pd.DataFrame(columns=["patient_name", "drug_name", "drug_code", "sig_text"])

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
                        "validation_decision",
                        "ai_validated_emoji",
                    ]
                )

            layout = _build_layout(input_df, output_df)
            live.update(layout, refresh=True)
            time.sleep(DISPLAY_REFRESH_SECONDS)


def _input_loop(input_path: Path, stop_event: threading.Event) -> None:
    """Foreground loop that lets you add rows while the tables are visible."""

    console.print(
        "\n[bold]Interactive input:[/bold] While the tables are shown above, you can "
        "add new prescriptions. These will be picked up by the left-hand "
        "pipeline the next time it processes rows from input_sigs.csv.\n"
    )

    while not stop_event.is_set():
        answer = console.input("Add a new prescription row? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            # Small pause so the Live display can repaint cleanly.
            time.sleep(0.2)
            continue

        if input_path.exists():
            df = pd.read_csv(input_path)
        else:
            df = pd.DataFrame(columns=["patient_name", "drug_name", "drug_code", "sig_text"])

        patient_name = console.input("  Patient name: ").strip()
        drug_name = console.input("  Drug name (as it appears in your DB): ").strip()
        drug_code = console.input("  Drug code (free text is fine): ").strip()
        sig_text = console.input("  Sig text (free-hand instructions): ").strip()

        new_row = {
            "patient_name": patient_name,
            "drug_name": drug_name,
            "drug_code": drug_code,
            "sig_text": sig_text,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(input_path, index=False)
        console.print("[green]  Added row to input table and saved to input_sigs.csv.[/green]\n")


def run_display() -> None:
    """Continuously display the input and output tables and accept new rows.

    Intended to run in the right-hand pane during the presentation.
    """

    input_path = DATA_DIR / "input_sigs.csv"
    output_path = DATA_DIR / "output_sigs.csv"

    if not input_path.exists():
        console.print(f"[red]Expected input file {input_path} to exist. Run `python scripts/setup_data.py` first.[/red]")
        return

    console.print("Starting live display. You can add new rows at any time below.\n")

    stop_event = threading.Event()
    display_thread = threading.Thread(
        target=_display_loop, args=(input_path, output_path, stop_event), daemon=True
    )
    display_thread.start()

    try:
        _input_loop(input_path, stop_event)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping live display...[/yellow]")
        stop_event.set()
        display_thread.join(timeout=1.0)
