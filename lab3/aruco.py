from typing import NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from lab3.config import CameraIntrinsics, Lab3CameraConfig, DEFAULT_LAB3_CAMERA
from lab3.geometry import rvec_tvec_to_T_cam_marker


class DetectedMarker(NamedTuple):
    marker_id: int
    T_cam_marker: npt.NDArray[np.float64]


class ArucoDetectionResult(NamedTuple):
    markers: list[DetectedMarker]
    corners: list[npt.NDArray[np.float32]]
    ids: npt.NDArray[np.int32] | None
    rvecs: npt.NDArray[np.float64]
    tvecs: npt.NDArray[np.float64]


class ArucoDetector:
    # Detects ArUco markers (DICT_ARUCO_ORIGINAL) and estimates pose relative to camera.

    def __init__(
        self,
        camera: Lab3CameraConfig | None = None,
        aruco_dictionary_id: int = cv2.aruco.DICT_ARUCO_ORIGINAL,
    ) -> None:
        self._camera = camera or DEFAULT_LAB3_CAMERA
        self._dictionary = cv2.aruco.getPredefinedDictionary(aruco_dictionary_id)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)

    @property
    def camera(self) -> Lab3CameraConfig:
        return self._camera

    def detect_full(self, frame: npt.NDArray[np.uint8]) -> ArucoDetectionResult:
        # Full detection: find markers, estimate poses, return everything for overlays/reports
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return ArucoDetectionResult(
                markers=[], corners=[], ids=None,
                rvecs=np.empty((0, 1, 3), dtype=np.float64),
                tvecs=np.empty((0, 1, 3), dtype=np.float64),
            )

        K = self._camera.intrinsics.camera_matrix
        dist = self._camera.intrinsics.dist_coeffs

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self._camera.marker_side_length_m, K, dist,
        )

        markers: list[DetectedMarker] = []
        for i in range(len(ids)):
            mid = int(ids[i][0])
            T = rvec_tvec_to_T_cam_marker(rvecs[i].reshape(3, 1), tvecs[i].reshape(3, 1))
            markers.append(DetectedMarker(mid, T))

        markers.sort(key=lambda m: m.marker_id)
        return ArucoDetectionResult(markers=markers, corners=corners, ids=ids, rvecs=rvecs, tvecs=tvecs)

    def find_tags(self, frame: npt.NDArray[np.uint8]) -> list[DetectedMarker]:
        # Detect visible ArUco markers and return (id, T_cam_marker) for each
        return self.detect_full(frame).markers
