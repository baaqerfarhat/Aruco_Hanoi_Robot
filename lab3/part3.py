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

    def TowerOfHanoi(self, n, fromRod, toRod, auxRod):
        if n == 0:
            return
        yield from self.TowerOfHanoi(n - 1, fromRod, auxRod, toRod)
        yield (fromRod, toRod)
        yield from self.TowerOfHanoi(n - 1, auxRod, toRod, fromRod)

if __name__ == "__main__":
    n = 3

    # A, C, B are the name of rods
    solver = HanoiSolver()
    solver.TowerOfHanoi(n, 'A', 'C', 'B')

if __name__ == "__main__":
    n = 3
    for fromRod, toRod in solver.TowerOfHanoi(n, 'A', 'C', 'B'):
        print(f"Move disk from {fromRod} to {toRod}")