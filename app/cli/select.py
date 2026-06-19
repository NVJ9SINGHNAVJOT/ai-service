"""
Tiny zero-dependency arrow-key selector for the terminal.

Renders a list of rows with a moving highlight and returns the chosen index.
Uses stdlib ``termios``/``tty`` for raw key input and ``rich`` for rendering,
so it adds no new dependencies. POSIX-only, which fits this Apple-Silicon
project. When stdin is not a TTY (piped input, CI) it falls back to a simple
numbered prompt.
"""

from __future__ import annotations

import sys
import termios
import tty
from typing import Optional, Sequence

from rich.console import Console, Group
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text

console = Console()


def select_from_list(
    rows: Sequence[Sequence[str]],
    title: str,
    columns: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """
    Show an interactive single-select menu and return the chosen row index.

    ``rows`` is a sequence of columns per row (e.g. ``(name, state, modalities)``).
    Returns the selected index, or ``None`` if the user cancels.
    """
    if not rows:
        return None

    if not sys.stdin.isatty():
        return _select_fallback(rows, title)

    index = 0
    with Live(
        _render(rows, index, title, columns),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            key = _read_key()
            if key in ("up", "down"):
                step = -1 if key == "up" else 1
                index = (index + step) % len(rows)
                live.update(_render(rows, index, title, columns))
                live.refresh()
            elif key == "enter":
                return index
            elif key == "cancel":
                return None


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render(
    rows: Sequence[Sequence[str]],
    index: int,
    title: str,
    columns: Optional[Sequence[str]],
) -> Group:
    """Build the renderable menu with the current row highlighted."""
    widths = _column_widths(rows, columns)

    lines: list[Text] = [
        Text(title, style="bold green"),
        Text("↑/↓ move · Enter select · q cancel", style="dim"),
        Text(""),
    ]
    if columns:
        header = "  " + "  ".join(col.ljust(w) for col, w in zip(columns, widths))
        lines.append(Text(header, style="dim"))

    for i, row in enumerate(rows):
        cells = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
        if i == index:
            lines.append(Text(f"▶ {cells}", style="bold cyan reverse"))
        else:
            lines.append(Text(f"  {cells}", style="white"))

    return Group(*lines)


def _column_widths(
    rows: Sequence[Sequence[str]],
    columns: Optional[Sequence[str]],
) -> list[int]:
    """Compute per-column widths so cells line up."""
    n_cols = max(len(r) for r in rows)
    widths = [0] * n_cols
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    if columns:
        for i, col in enumerate(columns[:n_cols]):
            widths[i] = max(widths[i], len(col))
    return widths


# ── Input ─────────────────────────────────────────────────────────────────────

def _read_key() -> str:
    """Read one keypress and map it to 'up' / 'down' / 'enter' / 'cancel'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape sequence (arrow keys) or a bare Esc
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "cancel"
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("k", "K"):
            return "up"
        if ch in ("j", "J"):
            return "down"
        if ch in ("q", "Q", "\x03"):  # q or Ctrl-C
            return "cancel"
        return "other"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _select_fallback(rows: Sequence[Sequence[str]], title: str) -> Optional[int]:
    """Numbered prompt used when stdin is not an interactive TTY."""
    console.print(f"[bold green]{title}[/bold green]")
    for i, row in enumerate(rows, start=1):
        console.print(f"  [cyan]{i}[/cyan]. " + "  ".join(str(c) for c in row))

    answer = Prompt.ask("Enter a number", default="").strip()
    if not answer.isdigit():
        return None
    choice = int(answer)
    if 1 <= choice <= len(rows):
        return choice - 1
    return None
