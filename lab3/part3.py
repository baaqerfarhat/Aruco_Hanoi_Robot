"""
Lab 3 Part 3 — Tower of Hanoi with in-hand camera and ArUco towers/blocks.

Wire in forward kinematics (classical DH), camera extrinsic :math:`T_5^{cam}` from the
handout, Part 1 detection, Part 2 grasps, and robot motion (URX / lab stack) inside
:class:`HanoiSolver`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Camera frame relative to link-5 frame (classical DH), prior to flange — from Lab 3 handout.
T_5_cam: npt.NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.1016],
        [0.0, 0.0, 1.0, 0.0848],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class HanoiSolver:
    """
    Orchestrate marker search, frame transforms, grasps, and moves for the Tower of Hanoi.

    Parameters
    ----------
    robot
        Optional robot interface (e.g. URX connection); ``None`` until you inject it in lab.
    """

    def __init__(self, robot: Any | None = None) -> None:
        self._robot = robot

    @property
    def robot(self) -> Any | None:
        return self._robot

    def marker_search(self) -> dict[int, npt.NDArray[np.float64]]:
        """
        Move the arm to view the table, detect ArUco tags, return marker id → :math:`T_{base}^{marker}`.

        Handout: find tower tags (IDs 0–2) and block tags (3–5); poses in base frame; search
        within the published workspace bounds.
        """
        raise NotImplementedError("Implement marker_search for Lab 3 Part 3.")

    def solve_tower_of_hanoi(self) -> None:
        """
        Plan and execute moves from tower 0 to tower 2 using Part 1 / Part 2 utilities.

        Use block widths (0.06, 0.10, 0.12 m) and grasp along x as recommended.
        """
        raise NotImplementedError("Implement solve_tower_of_hanoi for Lab 3 Part 3.")
