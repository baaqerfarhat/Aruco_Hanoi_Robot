"""Allow ``python -m lab3`` from the project directory."""

from __future__ import annotations

from lab3.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
