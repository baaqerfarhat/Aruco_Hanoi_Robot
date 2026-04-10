"""Rigid transforms and rotation helpers (camera ↔ marker)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def rvec_tvec_to_T_cam_marker(
    rvec: npt.NDArray[np.float64],
    tvec: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Build 4×4 T_cam_marker from OpenCV ``estimatePoseSingleMarkers`` outputs.

    Maps homogeneous points from marker frame to camera frame:
    ``p_c = R @ p_m + t`` (columns of R are marker axes expressed in the camera frame).
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T
