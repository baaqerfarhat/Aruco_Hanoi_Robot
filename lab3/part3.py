"""
Lab 3 Part 3 — Tower of Hanoi with in-hand camera and ArUco towers/blocks.

The camera is mounted on the robot near the **5th link**.  Its pose relative to frame 5
(classical DH) is given by :data:`T_5_cam`.  To get the camera pose in the **base frame**:

    ``T_base_cam = kin.fk_base_to_frame(q_deg, n_frames=5) @ T_5_cam``

That camera pose is combined with Part 1 detection outputs to obtain marker poses in the
base frame:

    ``T_base_marker = T_base_cam @ T_cam_marker``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from lab3.aruco import ArucoDetector
from lab3.part2 import get_grasp_pose
from lab3.ur10e_kinematics import (
    UR10eKinematics,
    inv_T,
    modified_joint_rad_to_classical_joint_deg,
    trans_z,
)

T_5_cam: npt.NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.1016],
        [0.0, 0.0, 1.0, 0.0848],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

BLOCK_WIDTHS_M: dict[int, float] = {3: 0.06, 4: 0.10, 5: 0.12}
TOWER_SIDE_M: float = 0.20

GRIPPER_TOOL_OFFSET = trans_z(0.20)

SEARCH_BOUNDS_MIN = np.array([-0.7, -1.0, -0.03], dtype=np.float64)
SEARCH_BOUNDS_MAX = np.array([0.7, 0.0, 0.03], dtype=np.float64)


class HanoiSolver:
    """
    Orchestrate marker search, frame transforms, grasps, and moves for the Tower of Hanoi.

    Parameters
    ----------
    robot
        Robot interface (e.g. URX connection); ``None`` until you inject it in lab.
    kin
        ``UR10eKinematics`` instance; created automatically if not provided.
    detector
        ``ArucoDetector`` instance; created automatically if not provided.
    """

    def __init__(
        self,
        robot: Any | None = None,
        kin: UR10eKinematics | None = None,
        detector: ArucoDetector | None = None,
    ) -> None:
        self._robot = robot
        self._kin = kin or UR10eKinematics()
        self._detector = detector or ArucoDetector()

    @property
    def robot(self) -> Any | None:
        return self._robot

    @property
    def kin(self) -> UR10eKinematics:
        return self._kin

    @property
    def detector(self) -> ArucoDetector:
        return self._detector

    def camera_pose_base(self, q_classical_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute :math:`T_{base}^{cam}` from the current joint angles.

        Uses FK to frame 5 (not 6!) and the known ``T_5_cam`` extrinsic.
        """
        T_base_5 = self._kin.fk_base_to_frame(q_classical_deg, n_frames=5)
        return T_base_5 @ T_5_cam

    def markers_in_base_frame(
        self,
        frame: npt.NDArray[np.uint8],
        q_classical_deg: npt.NDArray[np.float64],
    ) -> dict[int, npt.NDArray[np.float64]]:
        """
        Detect ArUco tags in ``frame`` and return marker_id → ``T_base_marker``.

        Combines ``T_base_cam`` (from FK to frame 5 + T_5_cam) with each ``T_cam_marker``
        detected by Part 1.
        """
        T_base_cam = self.camera_pose_base(q_classical_deg)
        detections = self._detector.find_tags(frame)
        return {m.marker_id: T_base_cam @ m.T_cam_marker for m in detections}

    def marker_search(self) -> dict[int, npt.NDArray[np.float64]]:
        """
        Move the arm to view the table, detect ArUco tags, return marker id → :math:`T_{base}^{marker}`.

        Handout: find tower tags (IDs 0–2) and block tags (3–5); poses in base frame; search
        within the published workspace bounds.
        """
        raise NotImplementedError(
            "Implement marker_search for Lab 3 Part 3.  "
            "Use self.markers_in_base_frame(frame, q_deg) at each viewpoint."
        )

    def TowerOfHanoi(self, n, fromRod, toRod, auxRod):
        if n == 0:
            return
        yield from self.TowerOfHanoi(n - 1, fromRod, auxRod, toRod)
        yield (fromRod, toRod)
        yield from self.TowerOfHanoi(n - 1, auxRod, toRod, fromRod)

    def solve_tower_of_hanoi(self) -> None:
        """
        Plan and execute moves from tower 0 to tower 2 using Part 1 / Part 2 utilities.

        Use block widths (0.06, 0.10, 0.12 m) and grasp along x as recommended.
        """
        raise NotImplementedError(
            "Implement solve_tower_of_hanoi for Lab 3 Part 3.  "
            "Use get_grasp_pose() and self.kin.ik() to compute joint targets."
        )


if __name__ == "__main__":
    n = 3
    solver = HanoiSolver()
    for fromRod, toRod in solver.TowerOfHanoi(n, 'A', 'C', 'B'):
        print(f"Move disk from {fromRod} to {toRod}")
