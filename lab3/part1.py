import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from lab3.aruco import ArucoDetector, ArucoDetectionResult, DetectedMarker
from lab3.aruco_viz import draw_aruco_overlay
from lab3.paths import ensure_annotated_images_dir, ensure_detection_reports_dir, repo_relative

logger = logging.getLogger(__name__)


class Part1PracticeRunner:
    # Loads test images, runs ArUco detection, saves annotated overlays and reports.

    def __init__(self, detector: ArucoDetector | None = None) -> None:
        self._detector = detector or ArucoDetector()

    @staticmethod
    def annotated_output_path(source_image: Path) -> Path:
        out_dir = ensure_annotated_images_dir()
        return out_dir / f"{source_image.stem}_annotated.png"

    @staticmethod
    def report_output_path(source_image: Path) -> Path:
        out_dir = ensure_detection_reports_dir()
        return out_dir / f"{source_image.stem}_report.txt"

    def find_markers_in_file(
        self,
        image_path: Path,
        *,
        save_annotated: bool = True,
    ) -> tuple[list[DetectedMarker] | None, str | None, Path | None, Path | None]:
        # Returns (markers, error, annotated_path, report_path)
        if not image_path.is_file():
            return None, f"File not found: {repo_relative(image_path)}", None, None

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            return None, f"Could not read image: {repo_relative(image_path)}", None, None

        detection = self._detector.detect_full(frame)
        markers = detection.markers

        annotated_path: Path | None = None
        report_path: Path | None = None
        if save_annotated:
            annotated_path = self._save_annotated_if_needed(image_path, frame, detection)
            report_path = self._save_report_file(image_path, markers, annotated_path)

        return markers, None, annotated_path, report_path

    def _save_annotated_if_needed(
        self,
        source_path: Path,
        frame_bgr: npt.NDArray[np.uint8],
        detection: ArucoDetectionResult,
    ) -> Path | None:
        cam = self._detector.camera.intrinsics
        axis_len = 0.5 * self._detector.camera.marker_side_length_m
        vis = draw_aruco_overlay(
            frame_bgr, detection, cam.camera_matrix, cam.dist_coeffs, axis_length_m=axis_len,
        )
        out = self.annotated_output_path(source_path)
        ok = cv2.imwrite(str(out), vis)
        if not ok:
            logger.warning("Failed to write annotated image: %s", repo_relative(out))
            return None
        return out

    def _save_report_file(
        self,
        source_path: Path,
        markers: list[DetectedMarker],
        annotated_path: Path | None,
    ) -> Path | None:
        out = self.report_output_path(source_path)
        text = self.format_detection_report(
            source_path, markers, annotated_path=annotated_path, report_path=out,
        )
        try:
            out.write_text(text, encoding="utf-8", newline="\n")
        except OSError as exc:
            logger.warning("Failed to write report file %s: %s", repo_relative(out), exc)
            return None
        return out

    @staticmethod
    def format_detection_report(
        image_path: Path,
        markers: list[DetectedMarker],
        *,
        annotated_path: Path | None = None,
        report_path: Path | None = None,
    ) -> str:
        lines: list[str] = [
            f"Detected {len(markers)} marker(s) in {image_path.name}",
            "",
        ]
        for m in markers:
            lines.extend(Part1PracticeRunner._format_marker_lines(m))
        if annotated_path is not None:
            lines.extend(["", f"Annotated image saved: {repo_relative(annotated_path)}", ""])
        if report_path is not None:
            lines.extend(["", f"Report saved: {repo_relative(report_path)}", ""])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _format_marker_lines(m: DetectedMarker) -> list[str]:
        T: npt.NDArray[np.float64] = m.T_cam_marker
        t = T[:3, 3]
        return [
            f"  ID {m.marker_id}: translation (camera frame) [m] = "
            f"({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})",
            f"           T_cam_marker =",
            f"{T}",
            "",
        ]

    def run(self, image_path: Path) -> int:
        # CLI entry point. Returns 0 on success, 1 on error.
        markers, err, ann, rep = self.find_markers_in_file(image_path, save_annotated=True)
        if err is not None:
            logger.error("%s", err)
            print(err, file=sys.stderr)
            if "not found" in err.lower():
                print(
                    "Add images under test_images/ in the repo, or pass --image PATH_TO_PNG.",
                    file=sys.stderr,
                )
            return 1

        assert markers is not None
        text = self.format_detection_report(image_path, markers, annotated_path=ann, report_path=rep)
        print(text, end="")
        return 0
