"""
UR10e forward and inverse kinematics (classical DH FK; analytical IK in modified DH).

Conventions
-----------
* :meth:`UR10eKinematics.fk` expects **classical** DH joint angles in **degrees** (same as
  the MPC ``UR10e.FK`` call path after ``DHModifiedToClassical`` → ``rad2deg``).
* :meth:`UR10eKinematics.ik` returns **modified** DH joint angles in **radians** (same as
  ``UR10e.IK`` in MPC). Convert to classical degrees for FK with
  :func:`modified_joint_rad_to_classical_joint_deg`.
* ``T_base_tool`` for IK is the homogeneous transform from the **robot base** to the
  **tool point**, consistent with ``inv(Tb0) @ T @ inv(T_tool)`` inside IK.

Optional tool transform ``Ttp_pen`` (MPC naming): applied on the flange side; use
``None`` to append no extra transform beyond the last DH row / default chain end.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt


def rot_x(a_rad: float) -> npt.NDArray[np.float64]:
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, ca, -sa, 0.0], [0.0, sa, ca, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rot_z(t_rad: float) -> npt.NDArray[np.float64]:
    ct, st = np.cos(t_rad), np.sin(t_rad)
    return np.array(
        [[ct, -st, 0.0, 0.0], [st, ct, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def trans_x(a_m: float) -> npt.NDArray[np.float64]:
    return np.array(
        [[1.0, 0.0, 0.0, a_m], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def trans_z(d_m: float) -> npt.NDArray[np.float64]:
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, d_m], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def dh_classical_tf(
    a_m: float, alpha_rad: float, d_m: float, theta_rad: float
) -> npt.NDArray[np.float64]:
    return rot_z(theta_rad) @ trans_z(d_m) @ trans_x(a_m) @ rot_x(alpha_rad)


def rotvec_to_R(rvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rvec / th
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def R_to_rotvec(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    c = float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3, dtype=np.float64)
    v = (
        np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=np.float64,
        )
        / (2.0 * np.sin(th))
    )
    return v * th


def pose6_to_T(pose6: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``[x, y, z, rx, ry, rz]`` with rotation as angle-axis (rotation vector)."""
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    p = pose6[:3]
    rvec = pose6[3:]
    R = rotvec_to_R(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def pose6_to_T_euler(pose6: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """[x, y, z, roll, pitch, yaw] -> 4x4 homogeneous transform.

    Uses roll-pitch-yaw Euler angles with the common convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    That means:
      - roll  rotates about x
      - pitch rotates about y
      - yaw   rotates about z
    """
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    x, y, z, roll, pitch, yaw = pose6

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr,  cr],
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp],
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1],
    ], dtype=np.float64)

    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T


def T_to_pose6(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = T[:3, 3]
    R = T[:3, :3]
    rvec = R_to_rotvec(R)
    return np.array([p[0], p[1], p[2], rvec[0], rvec[1], rvec[2]], dtype=np.float64)


def inv_T(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


class DHParameters(TypedDict):
    theta: list[float] | npt.NDArray[np.float64]
    d: list[float] | npt.NDArray[np.float64]
    a: list[float] | npt.NDArray[np.float64]
    alpha: list[float] | npt.NDArray[np.float64]


class ModifiedDHParameters(TypedDict):
    a_i_prev: list[float] | npt.NDArray[np.float64]
    alpha_i_prev: list[float] | npt.NDArray[np.float64]
    d_i: list[float] | npt.NDArray[np.float64]
    theta_i: list[float] | npt.NDArray[np.float64]


def dh_modified_to_classical_rad(theta_mod: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    theta_mod = np.asarray(theta_mod, dtype=np.float64).reshape(6)
    home_class = np.deg2rad(np.array([0.0, -90.0, 0.0, -90.0, 0.0, 180.0], dtype=np.float64))
    return theta_mod + home_class


def dh_classical_rad_to_modified_rad(theta_class: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    theta_class = np.asarray(theta_class, dtype=np.float64).reshape(6)
    home_class = np.deg2rad(np.array([0.0, -90.0, 0.0, -90.0, 0.0, 180.0], dtype=np.float64))
    return theta_class - home_class


def modified_joint_rad_to_classical_joint_deg(q_mod: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.rad2deg(dh_modified_to_classical_rad(q_mod))


class UR10eKinematics:

    def __init__(self) -> None:
        self.dof = 6
        self.Tb0 = trans_z(0.1807)
        self.T6tp = trans_z(0.11655)
        # TODO: THERE MIGHT BE NO ROTATION NEEDED IN THE TRANSFORM BELOW
        self.Ttp_gripper = pose6_to_T(np.array([0.0, 0.0, 0, 0.0, 0.0, np.pi/2], dtype=np.float64)) # rotation of gripper 90 degrees relative to flange

    def get_classical_dh_parameters(self, joint_angles_deg: npt.NDArray[np.float64]) -> DHParameters:
        alpha = [90.0, 0.0, 0.0, 90.0, -90.0, 0.0]
        a = [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0]
        d = [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655]
        theta = joint_angles_deg
        return {"theta": theta, "d": d, "a": a, "alpha": alpha}

    def get_modified_dh_parameters(
        self, joint_angles_deg: npt.NDArray[np.float64]
    ) -> ModifiedDHParameters:
        alpha_i_prev = [0.0, 90.0, 0.0, 0.0, -90.0, 90.0]
        a_i_prev = [0.0, 0.0, 0.6127, 0.57155, 0.0, 0.0]
        d_i = [0.0, 0.0, 0.0, 0.17415, 0.11985, 0.0]
        ja = np.asarray(joint_angles_deg, dtype=np.float64).reshape(6)
        theta_i = [
            ja[0],
            ja[1] + 90.0,
            ja[2],
            ja[3] - 90.0,
            ja[4],
            ja[5],
        ]
        return {
            "a_i_prev": a_i_prev,
            "alpha_i_prev": alpha_i_prev,
            "d_i": d_i,
            "theta_i": theta_i,
        }

    def DHModifiedToClassical(self, theta_mod: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return dh_modified_to_classical_rad(theta_mod)

    def DHClassicaltoModified(self, theta_class: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return dh_classical_rad_to_modified_rad(theta_class)

    def fk_base_to_frame(
        self,
        theta_classical_deg: npt.NDArray[np.float64],
        n_frames: int = 6,
    ) -> npt.NDArray[np.float64]:
        """
        FK chain through the first n_frames DH transforms (1–6)
        """
        if not 1 <= n_frames <= 6:
            raise ValueError(f"n_frames must be 1..6, got {n_frames}")
        theta_deg = np.asarray(theta_classical_deg, dtype=np.float64).reshape(6)
        dh = self.get_classical_dh_parameters(theta_deg)
        a = np.asarray(dh["a"], dtype=np.float64)
        d = np.asarray(dh["d"], dtype=np.float64)
        alpha_deg = dh["alpha"]
        theta_row = dh["theta"]
        T = np.eye(4, dtype=np.float64)
        for i in range(n_frames):
            T = T @ dh_classical_tf(
                float(a[i]),
                float(np.deg2rad(alpha_deg[i])),
                float(d[i]),
                float(np.deg2rad(theta_row[i])),
            )
        return T

    def fk(
        self,
        theta_classical_deg: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Forward kinematics: DH frame 0 → frame 6 (flange), with classical DH angles in degrees.
        """
        T = self.fk_base_to_frame(theta_classical_deg, n_frames=6)
        return T

    def ik(
        self,
        solution_type: str,
        T_base_gripper: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Inverse kinematics (analytical, modified DH convention).

        Returns one (6,) branch in **radians** (modified). ``solution_type`` is one of
        ``\"elbow_up\"``, ``\"elbow_down\"``, ``\"elbow_up_2\"``, ``\"elbow_down_2\"``.
        """
        dh_parameters = self.get_modified_dh_parameters(np.zeros(6, dtype=np.float64))
        a = np.asarray(dh_parameters["a_i_prev"], dtype=np.float64)
        d = np.asarray(dh_parameters["d_i"], dtype=np.float64)

        T_06 = np.linalg.inv(self.Tb0) @ T_base_gripper @ np.linalg.inv(self.Ttp_gripper) @ np.linalg.inv(self.T6tp)

        r11 = T_06[0, 0]
        r12 = T_06[0, 1]
        r13 = T_06[0, 2]
        r21 = T_06[1, 0]
        r22 = T_06[1, 1]
        r23 = T_06[1, 2]
        r31 = T_06[2, 0]
        r32 = T_06[2, 1]
        r33 = T_06[2, 2]
        px = T_06[0, 3]
        py = T_06[1, 3]
        pz = T_06[2, 3]

        E1, F1, G1 = py, -px, d[3]

        t1_pos = (-F1 + np.sqrt(E1**2 + F1**2 - G1**2)) / (G1 - E1)
        t1_neg = (-F1 - np.sqrt(E1**2 + F1**2 - G1**2)) / (G1 - E1)

        theta_1_1 = 2 * np.arctan(t1_pos)
        theta_1_2 = 2 * np.arctan(t1_neg)

        theta_6_1 = np.arctan2(
            r12 * np.sin(theta_1_1) - r22 * np.cos(theta_1_1),
            r21 * np.cos(theta_1_1) - r11 * np.sin(theta_1_1),
        )
        theta_6_2 = np.arctan2(
            r12 * np.sin(theta_1_2) - r22 * np.cos(theta_1_2),
            r21 * np.cos(theta_1_2) - r11 * np.sin(theta_1_2),
        )

        theta_5_1 = np.arctan2(
            (r21 * np.cos(theta_1_1) - r11 * np.sin(theta_1_1)) * np.cos(theta_6_1)
            + (r12 * np.sin(theta_1_1) - r22 * np.cos(theta_1_1)) * np.sin(theta_6_1),
            r13 * np.sin(theta_1_1) - r23 * np.cos(theta_1_1),
        )
        theta_5_2 = np.arctan2(
            (r21 * np.cos(theta_1_2) - r11 * np.sin(theta_1_2)) * np.cos(theta_6_2)
            + (r12 * np.sin(theta_1_2) - r22 * np.cos(theta_1_2)) * np.sin(theta_6_2),
            r13 * np.sin(theta_1_2) - r23 * np.cos(theta_1_2),
        )

        A1 = (r31 * np.cos(theta_6_1) - r32 * np.sin(theta_6_1)) / np.cos(theta_5_1)
        A2 = (r31 * np.cos(theta_6_2) - r32 * np.sin(theta_6_2)) / np.cos(theta_5_2)
        B1 = r31 * np.sin(theta_6_1) + r32 * np.cos(theta_6_1)
        B2 = r31 * np.sin(theta_6_2) + r32 * np.cos(theta_6_2)

        k1 = -px * np.cos(theta_1_1) - py * np.sin(theta_1_1) - d[4] * A1
        k2 = -px * np.cos(theta_1_2) - py * np.sin(theta_1_2) - d[4] * A2
        b1 = pz - d[4] * B1
        b2 = pz - d[4] * B2

        E2_1 = -2 * a[2] * b1
        E2_2 = -2 * a[2] * b2
        F2_1 = -2 * a[2] * k1
        F2_2 = -2 * a[2] * k2
        G2_1 = a[2] ** 2 + k1**2 + b1**2 - a[3] ** 2
        G2_2 = a[2] ** 2 + k2**2 + b2**2 - a[3] ** 2

        t2_pos_1 = (-F2_1 + np.sqrt(F2_1**2 + E2_1**2 - G2_1**2)) / (G2_1 - E2_1)
        t2_neg_1 = (-F2_1 - np.sqrt(F2_1**2 + E2_1**2 - G2_1**2)) / (G2_1 - E2_1)
        t2_pos_2 = (-F2_2 + np.sqrt(F2_2**2 + E2_2**2 - G2_2**2)) / (G2_2 - E2_2)
        t2_neg_2 = (-F2_2 - np.sqrt(F2_2**2 + E2_2**2 - G2_2**2)) / (G2_2 - E2_2)

        theta_2_1 = 2 * np.arctan(t2_pos_1)
        theta_2_2 = 2 * np.arctan(t2_neg_1)
        theta_2_3 = 2 * np.arctan(t2_pos_2)
        theta_2_4 = 2 * np.arctan(t2_neg_2)

        theta_3_1 = np.arctan2(k1 - a[2] * np.sin(theta_2_1), b1 - a[2] * np.cos(theta_2_1)) - theta_2_1
        theta_3_2 = np.arctan2(k1 - a[2] * np.sin(theta_2_2), b1 - a[2] * np.cos(theta_2_2)) - theta_2_2
        theta_3_3 = np.arctan2(k2 - a[2] * np.sin(theta_2_3), b2 - a[2] * np.cos(theta_2_3)) - theta_2_3
        theta_3_4 = np.arctan2(k2 - a[2] * np.sin(theta_2_4), b2 - a[2] * np.cos(theta_2_4)) - theta_2_4

        theta_4_1 = np.arctan2(A1, B1) - theta_2_1 - theta_3_1
        theta_4_2 = np.arctan2(A1, B1) - theta_2_2 - theta_3_2
        theta_4_3 = np.arctan2(A2, B2) - theta_2_3 - theta_3_3
        theta_4_4 = np.arctan2(A2, B2) - theta_2_4 - theta_3_4

        sol0 = np.array([theta_1_1, theta_2_1, theta_3_1, theta_4_1, theta_5_1, theta_6_1])
        sol1 = np.array([theta_1_1, theta_2_2, theta_3_2, theta_4_2, theta_5_1, theta_6_1])
        sol2 = np.array([theta_1_2, theta_2_3, theta_3_3, theta_4_3, theta_5_2, theta_6_2])
        sol3 = np.array([theta_1_2, theta_2_4, theta_3_4, theta_4_4, theta_5_2, theta_6_2])

        if solution_type == "elbow_up":
            return sol0
        if solution_type == "elbow_down":
            return sol1
        if solution_type == "elbow_up_2":
            return sol2
        if solution_type == "elbow_down_2":
            return sol3
        raise ValueError(
            f"Unknown solution_type {solution_type!r}; expected "
            "'elbow_up', 'elbow_down', 'elbow_up_2', or 'elbow_down_2'."
        )
