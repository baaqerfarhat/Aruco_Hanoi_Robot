import cv2
import numpy as np
import numpy.typing as npt

from lab3.aruco import ArucoDetectionResult


def draw_aruco_overlay(
    bgr: npt.NDArray[np.uint8],
    detection: ArucoDetectionResult,
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    *,
    axis_length_m: float,
    axis_thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    # Draw marker outlines and RGB pose axes on a copy of the image
    vis = bgr.copy()
    if detection.ids is None or len(detection.ids) == 0:
        return vis

    cv2.aruco.drawDetectedMarkers(vis, detection.corners, detection.ids)
    n = len(detection.ids)
    for i in range(n):
        cv2.drawFrameAxes(
            vis, camera_matrix, dist_coeffs,
            detection.rvecs[i], detection.tvecs[i],
            float(axis_length_m), axis_thickness,
        )
    return vis
