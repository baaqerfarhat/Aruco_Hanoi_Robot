"""
Lab 3 Part 2 — parallel-jaw grasping from an ArUco-tagged prism.

Implement ``get_grasp_pose`` per the handout: ArUco in the bottom-left corner of the prism,
block center in the +x / +y quadrant of the marker frame, prism thickness ~5 cm, gripper
center 20 cm along z from the last classical-DH frame, correct end-effector orientation.

Import FK/IK helpers from your earlier labs (e.g. Lab 1 URX utilities) here when needed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Nominal prism thickness [m] (handout ~5 cm).
PRISM_THICKNESS_M: float = 0.05

# Distance from last classical-DH frame to gripper center along that frame’s z-axis [m].
GRIPPER_OFFSET_FROM_LAST_FRAME_M: float = 0.20


def get_grasp_pose(
    T_base_aruco: npt.NDArray[np.float64],
    prism_width_m: float,
) -> npt.NDArray[np.float64]:
    """
    Compute :math:`T_{base}^{gripper}` (4×4) for an open parallel-jaw gripper ready to close on the prism.

    Parameters
    ----------
    T_base_aruco
        Homogeneous transform of the ArUco marker frame relative to the robot base (4×4).
    prism_width_m
        Width of the rectangular prism the tag is attached to [m].

    Returns
    -------
    ndarray
        :math:`T_{base}^{gripper}` (4×4, float64): base → gripper center in the open grasp pose.

    Notes
    -----
    Handout geometry: tag at bottom-left; prism center lies in the +x, +y quadrant of the
    marker frame; account for required gripper orientation and pre-grasp standoff from
    Part 2 pre-lab.
    """
    _ = (T_base_aruco, prism_width_m)
    raise NotImplementedError("Implement get_grasp_pose for Lab 3 Part 2.")
