"""
Lab 3 Part 2 — parallel-jaw grasping from an ArUco-tagged prism.

Geometry (from the handout):
  * ArUco tag at the **bottom-left** corner of the prism → block center is at
    ``(+width/2, +width/2, 0)`` in the marker frame.
  * Prisms have a thickness of ~5 cm.
  * The gripper center is 20 cm along z from the last classical-DH frame (tool offset used
    when calling IK: ``Ttp_pen = trans_z(0.20)``).
  * The gripper approaches from above with its z-axis pointing **down** (i.e. opposite the
    marker z-axis that points out of the tag surface).

Robot kinematics live in :mod:`lab3.ur10e_kinematics`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from lab3.ur10e_kinematics import pose6_to_T_euler, trans_z, rot_x
from lab3.part1 import Part1PracticeRunner
from lab3.paths import PRACTICE_IMAGE_NAME, default_practice_image_path

PRISM_THICKNESS_M: float = 0.05
GRIPPER_OFFSET_FROM_LAST_FRAME_M: float = 0.2098 # distance from last classical-DH frame to gripper center along z, in meters
W_MAX = 0.14 # maximum width of the gripper jaws in meters
DELTA_Z_MAX_GRIPPER = 0.0235 # maximum z displacement in gripper frame from fully open to fully closed, in meters


def get_grasp_pose(
    T_base_aruco: npt.NDArray[np.float64],
    prism_width_m: float,
) -> npt.NDArray[np.float64]:
    """
    Compute :math:`T_{base}^{gripper}` (4×4) for a top-down parallel-jaw grasp of the prism.

    Parameters
    ----------
    T_base_aruco
        Homogeneous transform of the ArUco marker frame relative to the robot base (4×4).
    prism_width_m
        Width of the rectangular prism the tag is attached to [m].

    Returns
    -------
    ndarray
        4×4 ``T_base_gripper`` placing the gripper center at the block center,
        oriented for a top-down grasp (gripper z pointing down, jaws along marker x).
    """
    T_base_aruco = np.asarray(T_base_aruco, dtype=np.float64).reshape(4, 4)
    T_base_box_bottom_center = T_base_aruco @ pose6_to_T_euler(np.array([prism_width_m / 2.0, prism_width_m / 2.0,
                                                                   -PRISM_THICKNESS_M, 0, 0, 0]))

    p = calculate_gripper_percentage(prism_width_m)
    delta_z_tip = p * DELTA_Z_MAX_GRIPPER / 100.0
    T_base_gripper_open = T_base_box_bottom_center @ trans_z(GRIPPER_OFFSET_FROM_LAST_FRAME_M + delta_z_tip) @ rot_x(np.pi)
    return T_base_gripper_open


def calculate_gripper_percentage(
    prism_width_m: float,
) -> float:
    """
    Calculate the percentage that the gripper should be closed based on the prism width.

    Parameters
    ----------
    prism_width_m
        Width of the rectangular prism the tag is attached to [m].

    Returns
    -------
    float
        Percentage (0-100) that the gripper should be closed, where 0% is fully open and 100% is fully closed.
    """
    if prism_width_m >= W_MAX:
        return 0.0
    elif prism_width_m <= 0.0:
        return 100.0
    else:
        return 100.0 * (W_MAX - prism_width_m) / W_MAX


if __name__ == "__main__":
    practice_runner = Part1PracticeRunner()
    image_path = default_practice_image_path()
    markers, error, annotated_path, report_path = practice_runner.find_markers_in_file(image_path, save_annotated=False)

    T_base_camera = pose6_to_T_euler(np.array([0.0, -0.4, 0.4, np.pi, 0, np.pi], dtype=np.float64))
    practice_width = 0.1

    for marker in markers or []:
        print(f"Marker ID {marker.marker_id}:")
        T_base_marker = T_base_camera @ marker.T_cam_marker
        print(f"T_base_marker:\n{T_base_marker}")
        T_base_gripper = get_grasp_pose(T_base_marker, practice_width)
        print(f"T_base_gripper:\n{T_base_gripper}")

    if error is not None:
        print(f"Error: {error}")