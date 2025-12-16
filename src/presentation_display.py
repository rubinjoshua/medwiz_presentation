from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table

from .config import DATA_DIR, DISPLAY_REFRESH_SECONDS

console = Console()


def _with_row_count_suffix(title: str, shown: int, total: int) -> str:
    if total <= shown:
        return title
    return f"{title} (showing last {shown} / {total})"


def _make_translation_table(df: pd.DataFrame, *, shown_rows: int | None = None, total_rows: int | None = None) -> Table:
    title = "AI translations (pre-validation)"
    if shown_rows is not None and total_rows is not None:
        title = _with_row_count_suffix(title, shown_rows, total_rows)

    table = Table(title=title, expand=True)

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


def _make_output_table(df: pd.DataFrame, *, shown_rows: int | None = None, total_rows: int | None = None) -> Table:
    title = "AI translated + validated prescriptions"
    if shown_rows is not None and total_rows is not None:
        title = _with_row_count_suffix(title, shown_rows, total_rows)

    table = Table(title=title)

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


def _fit_df_to_height(
    *,
    df: pd.DataFrame,
    make_table: Callable[[pd.DataFrame, int, int], Table],
    target_height: int,
    target_width: int,
) -> tuple[pd.DataFrame, int, int]:
    """Return a tail-sliced DF that renders within target_height.

    We do this so the output appears to "auto-scroll" in Rich's full-screen Live mode
    (screen=True), where you otherwise can't scroll back.

    Notes:
    - We render tables in-memory using console.render_lines to measure the height.
    - This keeps the table header visible because each refresh re-renders the header
      and only shows the last N rows.
    """

    total = len(df)
    if total == 0:
        return df, 0, 0

    # Fast-path: everything fits.
    options = console.options.update(width=target_width)
    if len(console.render_lines(make_table(df, total, total), options=options)) <= target_height:
        return df, total, total

    # Binary search: find the largest tail(n) that fits.
    low = 0
    high = total
    best = 0

    while low <= high:
        mid = (low + high) // 2
        sliced = df.tail(mid)
        rendered_height = len(console.render_lines(make_table(sliced, mid, total), options=options))

        if rendered_height <= target_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return df.tail(best), best, total


def _build_layout(
    *,
    translated_df: pd.DataFrame,
    output_df: pd.DataFrame,
    console_width: int,
    console_height: int,
) -> Layout:
    # Split the screen roughly in half.
    top_height = max(1, console_height // 2)
    bottom_height = max(1, console_height - top_height)

    # Keep a little slack so borders/wrapping don't overflow the region.
    # (We'd rather show fewer rows than spill off-screen.)
    slack = 1

    fitted_translated_df, shown_t, total_t = _fit_df_to_height(
        df=translated_df,
        make_table=lambda d, shown, total: _make_translation_table(d, shown_rows=shown, total_rows=total),
        target_height=max(1, top_height - slack),
        target_width=console_width,
    )
    fitted_output_df, shown_o, total_o = _fit_df_to_height(
        df=output_df,
        make_table=lambda d, shown, total: _make_output_table(d, shown_rows=shown, total_rows=total),
        target_height=max(1, bottom_height - slack),
        target_width=console_width,
    )

    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=1),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].update(_make_translation_table(fitted_translated_df, shown_rows=shown_t, total_rows=total_t))
    layout["bottom"].update(_make_output_table(fitted_output_df, shown_rows=shown_o, total_rows=total_o))
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

            size = live.console.size
            layout = _build_layout(
                translated_df=translated_df,
                output_df=output_df,
                console_width=size.width,
                console_height=size.height,
            )
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
