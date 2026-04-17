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

PRISM_THICKNESS_M: float = 0.05
GRIPPER_OFFSET_FROM_LAST_FRAME_M: float = 0.20


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
    T = np.asarray(T_base_aruco, dtype=np.float64).reshape(4, 4)
    R_marker = T[:3, :3]

    p_block_marker = np.array(
        [prism_width_m / 2.0, prism_width_m / 2.0, -PRISM_THICKNESS_M / 2.0, 1.0],
        dtype=np.float64,
    )
    p_block_base = (T @ p_block_marker)[:3]

    # Rotate 180° about the marker x-axis so gripper z points down (into the block).
    #   Rx(π) = diag(1, -1, -1)
    # Resulting columns: x_g = marker_x, y_g = -marker_y, z_g = -marker_z.
    R_grasp = R_marker @ np.diag([1.0, -1.0, -1.0])

    T_base_gripper = np.eye(4, dtype=np.float64)
    T_base_gripper[:3, :3] = R_grasp
    T_base_gripper[:3, 3] = p_block_base
    return T_base_gripper
