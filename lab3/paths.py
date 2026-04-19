from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
PRACTICE_IMAGE_NAME = "aruco detection test practice.png"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_relative(path: Path | str, *, root: Path | None = None) -> str:
    # Pretty path relative to repo root, forward slashes for cross-platform consistency
    root = (root or project_root()).resolve()
    p = Path(path).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def test_images_dir() -> Path:
    return project_root() / "test_images"


def ensure_test_images_dir() -> Path:
    d = test_images_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def annotated_images_dir() -> Path:
    return test_images_dir() / "annotated"


def ensure_annotated_images_dir() -> Path:
    d = annotated_images_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def detection_reports_dir() -> Path:
    return test_images_dir() / "reports"


def ensure_detection_reports_dir() -> Path:
    d = detection_reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_test_images() -> list[Path]:
    # Sorted image files under test_images/ (skips annotated outputs)
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
    # Prefer handout practice image, else first image in test_images/
    ensure_test_images_dir()
    preferred = test_images_dir() / PRACTICE_IMAGE_NAME
    if preferred.is_file():
        return preferred
    listed = list_test_images()
    if listed:
        return listed[0]
    return project_root() / PRACTICE_IMAGE_NAME
