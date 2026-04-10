"""Tkinter GUI for Lab 3 (Parts 1–3)."""

from __future__ import annotations

__all__ = ["run_app"]


def run_app() -> None:
    """Start the Lab 3 desktop application (blocks until the window is closed)."""
    from lab3.gui.main_window import main

    main()
