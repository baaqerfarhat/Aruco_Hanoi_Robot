from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from pyrobotiqur import RobotiqGripper

from lab3.aruco import ArucoDetector
from lab3.part2 import get_grasp_pose, calculate_gripper_percentage
from lab3.ur10e_kinematics import (
    UR10eKinematics,
    inv_T,
    modified_joint_rad_to_classical_joint_deg,
    trans_z,
    rot_x,
    pose6_to_T_euler
)

UR_IP = "192.168.0.2"

BLOCK_HEIGHT_M: float = 0.05

# Camera extrinsic relative to link 5
T_5_cam: npt.NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.1016],
        [0.0, 0.0, 1.0, 0.0848],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Block marker IDs -> widths in meters
BLOCK_WIDTHS_M: dict[int, float] = {3: 0.06, 4: 0.10, 5: 0.12}
TOWER_SIDE_M: float = 0.20

# Workspace bounds for marker search grid
SEARCH_BOUNDS_MIN = np.array([-0.7, -1.0, -0.03], dtype=np.float64)
SEARCH_BOUNDS_MAX = np.array([0.7, 0.0, 0.03], dtype=np.float64)


class HanoiSolver:

    def __init__(
        self,
        robot: Any | None = None,
        gripper: RobotiqGripper | None = None,
        kin: UR10eKinematics | None = None,
        detector: ArucoDetector | None = None,
        camera_source: int | str | None = 0,
    ) -> None:
        self._robot = robot
        self._gripper = gripper
        self._kin = kin or UR10eKinematics()
        self._detector = detector or ArucoDetector()
        self._camera_source = camera_source
        self._cap: cv2.VideoCapture | None = None
        if camera_source is not None:
            self._cap = cv2.VideoCapture(camera_source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Unable to open camera source {camera_source!r}")

        # Hanoi disk number -> ArUco marker ID
        self.hanoi_disk_to_marker_id = {1: 3, 2: 4, 3: 5}

        # Track stacking height per tower (tower 0 starts with all 3 blocks)
        self.tower_placement_heights = {
            0: 0.01 + 3 * BLOCK_HEIGHT_M,
            1: 0.01,
            2: 0.01,
        }

    def capture_frame(self) -> npt.NDArray[np.uint8] | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def camera_pose_base(self, q_classical_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # T_base_cam = FK_to_frame5 @ T_5_cam
        T_base_5 = self._kin.fk_base_to_frame(q_classical_deg, n_frames=5)
        return T_base_5 @ T_5_cam

    def markers_in_base_frame(
        self,
        frame: npt.NDArray[np.uint8],
        q_classical_deg: npt.NDArray[np.float64],
    ) -> dict[int, npt.NDArray[np.float64]]:
        # Detect markers in image, transform poses to robot base frame
        T_base_cam = self.camera_pose_base(q_classical_deg)
        detections = self._detector.find_tags(frame)
        return {m.marker_id: T_base_cam @ m.T_cam_marker for m in detections}

    def marker_search(self) -> dict[int, npt.NDArray[np.float64]]:
        # Sweep a 5x5 grid over the workspace, look down at each point, detect markers
        markers = {}
        x_grid = np.linspace(SEARCH_BOUNDS_MIN[0], SEARCH_BOUNDS_MAX[0], 5)
        y_grid = np.linspace(SEARCH_BOUNDS_MIN[1], SEARCH_BOUNDS_MAX[1], 5)
        z_view = 0.4

        for x in x_grid:
            for y in y_grid:
                # Camera looking straight down at (x, y, z_view)
                T_base_cam_desired = np.eye(4, dtype=np.float64) @ rot_x(np.pi)
                T_base_cam_desired[:3, 3] = [x, y, z_view]

                # Convert desired camera pose to gripper pose for IK
                T_base_5_desired = T_base_cam_desired @ inv_T(T_5_cam)
                T_base_gripper = T_base_5_desired @ self._kin.T6tp @ self._kin.Ttp_gripper

                q_rad = self._kin.ik("elbow_up", T_base_gripper)
                q_deg = modified_joint_rad_to_classical_joint_deg(q_rad)

                if self._robot is not None:
                    self._robot.movej(q_deg * np.pi / 180, acc=0.1, vel=0.1)

                frame = self.capture_frame()
                if frame is None:
                    continue

                detections = self.markers_in_base_frame(frame, q_deg)
                markers.update(detections)

        return markers

    def get_tower_centers(
        self,
        markers: dict[int, npt.NDArray[np.float64]]
    ) -> dict[int, npt.NDArray[np.float64]]:
        # Tower center = marker pose offset by half the tower side in x and y
        T_tower_centers = {}
        T_marker_to_tower_center = pose6_to_T_euler(
            np.array([TOWER_SIDE_M / 2.0, TOWER_SIDE_M / 2.0, 0.0, 0.0, 0.0, 0.0])
        )
        for marker_id in range(3):
            if marker_id in markers:
                T_tower_centers[marker_id] = markers[marker_id] @ T_marker_to_tower_center
        return T_tower_centers

    def TowerOfHanoi(self, n: int, fromRod: int, toRod: int, auxRod: int):
        # Recursive Tower of Hanoi generator: yields (disk_number, from_tower, to_tower)
        if n == 0:
            return
        yield from self.TowerOfHanoi(n - 1, fromRod, auxRod, toRod)
        yield (n, fromRod, toRod)
        yield from self.TowerOfHanoi(n - 1, auxRod, toRod, fromRod)

    def solve_tower_of_hanoi(self) -> None:
        n = 3
        for hanoi_disk, from_tower, to_tower in self.TowerOfHanoi(n, 0, 2, 1):
            block_id = self.hanoi_disk_to_marker_id[hanoi_disk]
            print(f"Move block with ID {block_id} from tower {from_tower} to tower {to_tower}")

            # Search for all markers
            markers = self.marker_search()
            print("Detected markers and their poses in base frame:")
            for marker_id, T_base_marker in markers.items():
                print(f"Marker ID {marker_id}: T_base_marker:\n{T_base_marker}")

            tower_centers = self.get_tower_centers(markers)
            print("\nEstimated tower centers in base frame:")
            for tower_id, T_tower_center in tower_centers.items():
                print(f"Tower {tower_id}: T_tower_center:\n{T_tower_center}")

            print(3 in list(markers.keys()))

            # Retry until we find the block and target tower
            while block_id not in list(markers.keys()) or to_tower not in list(tower_centers.keys()):
                print(f"Block with ID {block_id} or tower {to_tower} not found, retrying marker search...")
                markers = self.marker_search()
                tower_centers = self.get_tower_centers(markers)
                time.sleep(1.0)

            # Open gripper before approach
            if self._gripper is not None:
                self._gripper.move_percent(0)

            # Compute grasp pose and move to pick up
            T_grasp_pose = get_grasp_pose(T_base_marker=markers[block_id], prism_width_m=BLOCK_WIDTHS_M[block_id])
            q_rad = self._kin.ik("elbow_up", T_grasp_pose)
            q_deg = modified_joint_rad_to_classical_joint_deg(q_rad)

            if self._robot is not None:
                self._robot.movej(q_deg * np.pi / 180, acc=0.1, vel=0.1)

            # Close gripper to grasp
            if self._gripper is not None:
                self._gripper.move_percent(calculate_gripper_percentage(BLOCK_WIDTHS_M[block_id]))

            # Move to placement on target tower
            T_tower_placement = tower_centers[to_tower] @ trans_z(self.tower_placement_heights[to_tower])
            q_rad_place = self._kin.ik("elbow_up", T_tower_placement)
            q_deg_place = modified_joint_rad_to_classical_joint_deg(q_rad_place)
            if self._robot is not None:
                self._robot.movej(q_deg_place * np.pi / 180, acc=0.1, vel=0.1)

            # Open gripper to release
            if self._gripper is not None:
                self._gripper.move_percent(0)

            # Update stacking heights
            self.tower_placement_heights[to_tower] += BLOCK_HEIGHT_M
            self.tower_placement_heights[from_tower] -= BLOCK_HEIGHT_M


if __name__ == "__main__":
    # import urx
    # rob = urx.Robot(UR_IP)
    # rob.set_tcp((0, 0, 0, 0, 0, 0))
    # ...
    # g = RobotiqGripper(UR_IP, port=63352, timeout=20.0)
    # g.connect()
    # g.activate()

    rob = None   # <-- this was the active line
    g = None     # <-- this was the active line
    solver = HanoiSolver(robot=rob, gripper=g)
    solver.solve_tower_of_hanoi()