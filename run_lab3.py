"""
CDS/ME-235 Lab 3 — run the CLI or GUI from the project root.

This file is intentionally not named ``lab3.py`` because that would shadow the ``lab3/``
Python package in the same directory.

For Canvas, if the course asks for ``235lab3 <GROUP NAMES>.py``, copy or rename this file
to match (the contents stay the same).

Examples (run inside this project folder)::

    python run_lab3.py
    python run_lab3.py --gui
"""

from __future__ import annotations

from lab3.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
