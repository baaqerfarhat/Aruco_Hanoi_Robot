# Tower of Hanoi Solver — UR10e + In-Hand Camera

Autonomous Tower of Hanoi solver using a **UR10e** robot arm with an **in-hand camera** and **ArUco marker** detection. The robot scans the workspace, locates towers and blocks, computes grasp poses via forward/inverse kinematics, and executes pick-and-place moves to solve the puzzle.

---

## How it works

1. **Camera mounted on link 5** of the UR10e scans the table by moving through a grid of viewpoints
2. **ArUco markers** (IDs 0-2 = towers, IDs 3-5 = blocks) are detected at each viewpoint and transformed into the robot's base frame using FK
3. **Grasp poses** are computed for each block based on its marker position, block width, and gripper geometry
4. The **Tower of Hanoi algorithm** generates the move sequence (tower 0 -> tower 2), and the robot executes each pick-and-place using analytical IK

The entire pipeline — vision, kinematics, grasp planning, and game logic — runs through the GUI with live camera feed and real-time status.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python run_lab3.py --gui
```

### Connecting to the robot

1. Open the GUI and go to the **Part 3 — Tower of Hanoi** tab
2. Enter the robot IP (default `192.168.0.2`) and camera source
3. Check **"Connect to real robot + gripper"**
4. Click **Initialize solver** — this connects to the UR10e via URX, activates the Robotiq gripper, and opens the camera
5. Click **Solve Tower of Hanoi** — the robot will scan, locate everything, and play the game

Use **Stop Robot** to halt arm movement or **STOP ALL** for a full emergency stop (halts arm, opens gripper, kills the running task).

---

## Key components

### Forward & Inverse Kinematics (`ur10e_kinematics.py`)

- **Classical DH** forward kinematics — full 6-joint chain or partial (e.g. 5 frames for camera pose)
- **Analytical IK** in modified DH convention with 4 solution branches (elbow up/down x2)
- Joint angle conversion between modified DH (radians) and classical DH (degrees)

### ArUco Detection (`aruco.py`)

- Uses OpenCV's `DICT_ARUCO_ORIGINAL` dictionary
- Estimates 6-DOF marker pose via PnP with known camera intrinsics
- Returns `T_cam_marker` (4x4 homogeneous transform) for each detected tag

### Grasp Planning (`part2.py`)

- `get_grasp_pose(T_base_aruco, prism_width_m)` computes the gripper transform for a top-down parallel-jaw grasp
- Accounts for marker-to-block-center offset, prism thickness, gripper standoff, and jaw opening

### Tower of Hanoi Solver (`part3.py`)

- `HanoiSolver` orchestrates the full pipeline:
  - `marker_search()` — sweeps a 5x5 grid over the workspace, detects markers at each viewpoint, transforms poses to base frame
  - `get_tower_centers()` — computes tower center positions from tower marker poses
  - `TowerOfHanoi()` — recursive generator yielding (disk, from_tower, to_tower) moves
  - `solve_tower_of_hanoi()` — executes the game: search, pick, place, repeat for all 7 moves
- Camera extrinsic `T_5_cam` (camera relative to link 5) from the lab handout

### GUI (`gui/main_window.py`)

- Dark-themed interface (ttkbootstrap) with tabs for each lab part
- **Part 3 tab**: live camera feed, process status dashboard (status / step / markers / moves), log output
- Robot connection management, emergency stop controls

---

## Project layout

```
run_lab3.py              # Entry point (--gui or CLI)
lab3/
  ur10e_kinematics.py    # FK, IK, DH helpers, rigid transform utils
  aruco.py               # ArUco detection + pose estimation
  part2.py               # Grasp pose computation
  part3.py               # HanoiSolver (marker search, game logic, robot control)
  part1.py               # Practice image runner
  config.py              # Camera intrinsics from handout
  geometry.py            # rvec/tvec -> homogeneous transform
  gui/                   # Tkinter GUI
test_images/             # Drop test images here
```

---

## Requirements

- Python **3.10+**
- UR10e robot with URX (`pip install urx`)
- Robotiq gripper (`pip install pyrobotiqur`)
- OpenCV contrib (ArUco), NumPy, Pillow, ttkbootstrap

```bash
pip install -r requirements.txt
```

---

## Running without a robot

The GUI works in **dry-run mode** — leave the "Connect to real robot" checkbox unchecked. You can still test Part 1 (ArUco detection on images) and Part 2 (grasp pose computation) without any hardware. Part 3 will attempt the full pipeline but won't move anything without a robot connection.
