"""Camera intrinsics and scene constants for Lab 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CameraIntrinsics:
    """Brown–Conrady distortion model: 3×3 K and distortion coefficients (OpenCV layout)."""

    camera_matrix: npt.NDArray[np.float64]
    dist_coeffs: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_matrix", np.asarray(self.camera_matrix, dtype=np.float64))
        object.__setattr__(
            self,
            "dist_coeffs",
            np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1),
        )


@dataclass(frozen=True)
class Lab3CameraConfig:
    """Bundled lab camera parameters and default ArUco square size."""

    intrinsics: CameraIntrinsics
    marker_side_length_m: float

    @classmethod
    def from_lab3_handout(cls) -> Lab3CameraConfig:
        """Intrinsics and 2 cm tag size from the April 2026 Lab 3 v1 handout."""
        K = np.array(
            [
                [1698.75, 0.0, 1115.55],
                [0.0, 1695.98, 751.98],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        d = np.array(
            [-0.00670872, -0.1481124, -0.00250596, 0.00299921, -1.68711031],
            dtype=np.float64,
        ).reshape(-1, 1)
        return cls(
            intrinsics=CameraIntrinsics(camera_matrix=K, dist_coeffs=d),
            marker_side_length_m=0.02,
        )


DEFAULT_LAB3_CAMERA = Lab3CameraConfig.from_lab3_handout()
