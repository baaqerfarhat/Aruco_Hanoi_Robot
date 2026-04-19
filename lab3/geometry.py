from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def rvec_tvec_to_T_cam_marker(
    rvec: npt.NDArray[np.float64],
    tvec: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # Build 4x4 T_cam_marker from OpenCV estimatePoseSingleMarkers outputs
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T
