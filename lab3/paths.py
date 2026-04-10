"""
Filesystem paths shared by CLI and GUI.

All locations are derived from :func:`project_root`, i.e. the repository root (the directory
that contains the ``lab3`` package and ``test_images`` after a ``git clone``). Nothing is
tied to a specific machine, drive, or username.
"""

from __future__ import annotations

from pathlib import Path

# Images with these suffixes appear in the Part 1 GUI dropdown (project ``test_images/`` folder).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})

# Handout practice image name (place the file under ``test_images/``).
PRACTICE_IMAGE_NAME = "aruco detection test practice.png"


def project_root() -> Path:
    """
    Repository root: parent of the ``lab3`` package directory.

    Resolves from this file’s location so it works on any OS after cloning the repo.
    """
    return Path(__file__).resolve().parent.parent


def repo_relative(path: Path | str, *, root: Path | None = None) -> str:
    """
    Pretty path for logs, reports, and UI: relative to the repo root when possible.

    Uses forward slashes so output looks the same on Windows, macOS, and Linux. If ``path`` is
    outside the repository (e.g. a user-picked file elsewhere), returns that path with forward
    slashes (absolute), not a machine-specific Windows-only form.
    """
    root = (root or project_root()).resolve()
    p = Path(path).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def test_images_dir() -> Path:
    """Folder where GUI/CLI look for test images (add new shots here)."""
    return project_root() / "test_images"


def ensure_test_images_dir() -> Path:
    d = test_images_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def annotated_images_dir() -> Path:
    """Subfolder of ``test_images/`` where annotated outputs are written."""
    return test_images_dir() / "annotated"


def ensure_annotated_images_dir() -> Path:
    d = annotated_images_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def detection_reports_dir() -> Path:
    """Subfolder of ``test_images/`` where text reports are written (``<stem>_report.txt``)."""
    return test_images_dir() / "reports"


def ensure_detection_reports_dir() -> Path:
    d = detection_reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_test_images() -> list[Path]:
    """
    Sorted list of image files directly under ``test_images/`` (not subfolders).

    Skips names ending with ``_annotated`` before the extension so saved overlays do not
    clutter the Part 1 dropdown.
    """
    d = test_images_dir()
    if not d.is_dir():
        return []
    files: list[Path] = []
    for p in d.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem_lower = p.stem.lower()
        if stem_lower.endswith("_annotated") or stem_lower.endswith("-annotated"):
            continue
        files.append(p)
    files.sort(key=lambda p: p.name.lower())
    return files


def default_practice_image_path() -> Path:
    """
    Default image for CLI ``--image``.

    Prefer ``test_images/`` + :data:`PRACTICE_IMAGE_NAME`, else the first file in that folder,
    else the legacy path next to the submission script (older layouts).
    """
    ensure_test_images_dir()
    preferred = test_images_dir() / PRACTICE_IMAGE_NAME
    if preferred.is_file():
        return preferred
    listed = list_test_images()
    if listed:
        return listed[0]
    return project_root() / PRACTICE_IMAGE_NAME
