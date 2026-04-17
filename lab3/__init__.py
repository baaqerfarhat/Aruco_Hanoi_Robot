"""CDS/ME-235 Lab 3 — vision, grasping, and Tower of Hanoi (package root)."""

from lab3.aruco import ArucoDetector, DetectedMarker
from lab3.config import CameraIntrinsics, Lab3CameraConfig, DEFAULT_LAB3_CAMERA
from lab3.part2 import get_grasp_pose
from lab3.part3 import HanoiSolver, T_5_cam
from lab3.ur10e_kinematics import (
    UR10eKinematics,
    dh_classical_rad_to_modified_rad,
    dh_modified_to_classical_rad,
    inv_T,
    modified_joint_rad_to_classical_joint_deg,
    pose6_to_T,
    T_to_pose6,
    trans_z,
)

__all__ = [
    "ArucoDetector",
    "CameraIntrinsics",
    "DEFAULT_LAB3_CAMERA",
    "DetectedMarker",
    "Lab3CameraConfig",
    "get_grasp_pose",
    "HanoiSolver",
    "T_5_cam",
    "UR10eKinematics",
    "dh_classical_rad_to_modified_rad",
    "dh_modified_to_classical_rad",
    "inv_T",
    "modified_joint_rad_to_classical_joint_deg",
    "pose6_to_T",
    "T_to_pose6",
    "trans_z",
    "__version__",
]

__version__ = "0.1.0"
