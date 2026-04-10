"""CDS/ME-235 Lab 3 — vision, grasping, and Tower of Hanoi (package root)."""

from __future__ import annotations

from lab3.aruco import ArucoDetector, DetectedMarker
from lab3.config import CameraIntrinsics, Lab3CameraConfig, DEFAULT_LAB3_CAMERA
from lab3.part2 import get_grasp_pose
from lab3.part3 import HanoiSolver, T_5_cam

__all__ = [
    "ArucoDetector",
    "CameraIntrinsics",
    "DEFAULT_LAB3_CAMERA",
    "DetectedMarker",
    "Lab3CameraConfig",
    "get_grasp_pose",
    "HanoiSolver",
    "T_5_cam",
    "__version__",
]

__version__ = "0.1.0"
