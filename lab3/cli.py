"""Command-line interface for Lab 3 (argparse, logging, exit codes)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lab3 import __version__
from lab3.part1 import Part1PracticeRunner
from lab3.paths import default_practice_image_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab3",
        description="CDS/ME-235 Lab 3 — Part 1 ArUco practice (extend for Parts 2–3).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the graphical interface (tabs, status bar, Part 1 controls).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print debug messages on stderr.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=default_practice_image_path(),
        help="Input image (default: test_images/ handout name or first image in that folder).",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.gui:
        from lab3.gui import run_app

        run_app()
        return 0

    runner = Part1PracticeRunner()
    return runner.run(args.image)
