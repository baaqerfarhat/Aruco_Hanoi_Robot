# CDS/ME-235 Lab 3 — Vision, Grasping & Tower of Hanoi

Course project for **Caltech CDS/ME-235** (Spring 2026): integrate **ArUco** vision with a **UR10e**-class pipeline, practice **parallel-jaw** grasps, and (in Part 3) play **Tower of Hanoi** using in-hand camera data and frame transforms.

This repository is self-contained: clone it, install dependencies, and run from the project root (the folder that contains `lab3/` and `run_lab3.py`).

---

## What’s implemented

| Part | Topic | Status |
|------|--------|--------|
| **1** | ArUco detection, camera pose (`DICT_ARUCO_ORIGINAL`), intrinsics from the handout, GUI + CLI, annotated images & text reports | **Implemented** |
| **2** | Parallel-jaw grasp pose from ArUco + prism geometry (`get_grasp_pose`) | **Stub** — fill in `lab3/part2.py` |
| **3** | `HanoiSolver`: marker search, base-frame poses, moves with FK / in-hand camera (`T_5^{cam}`) | **Stub** — fill in `lab3/part3.py` |

Results and writeups for Parts 2 and 3 will be added here as the lab sections are completed.

---

## Requirements

- Python **3.10+**
- See `requirements.txt` / `pyproject.toml` (NumPy, OpenCV **contrib** for ArUco, Pillow for the GUI).

```bash
pip install -r requirements.txt
```

---

## How to run

From **this directory** (repo root):

```bash
# GUI (Part 1: pick test images under test_images/, run detection, view overlays)
python run_lab3.py --gui

# CLI — Part 1 on default or chosen image
python run_lab3.py
python run_lab3.py --image test_images/your_image.png
```

The launcher is named `run_lab3.py` (not `lab3.py`) so it does not shadow the `lab3/` package. If the course asks for a specific submission filename (e.g. `235lab3 <GROUP>.py`), copy or rename `run_lab3.py` accordingly.

---

## Project layout

- `run_lab3.py` — entry point (CLI / `--gui`)
- `lab3/` — package code  
  - `aruco.py`, `aruco_viz.py`, `config.py`, `geometry.py`, `paths.py` — Part 1  
  - `part1.py` — practice runner & saved outputs  
  - `part2.py` — grasp pose (**to implement**)  
  - `part3.py` — Hanoi solver (**to implement**)  
  - `gui/` — Tkinter UI  
- `test_images/` — drop input images here; outputs go to `test_images/annotated/` and `test_images/reports/`

---

## Part 2 (planned)

- **`get_grasp_pose(T_base_aruco, prism_width_m)`** → homogeneous `T_base_gripper` for an open gripper ready to close on a rectangular prism.  
- Geometry from the handout: tag on the bottom-left face, prism center in the +x/+y quadrant of the marker frame, nominal thickness, and gripper offset along the last classical-DH **z** axis.  
- Wire in **FK/IK** and Robotiq-style constraints from earlier labs as needed.

---

## Part 3 (planned)

- **`HanoiSolver`**: **`marker_search()`** (scan workspace, ArUco IDs for towers 0–2 and blocks 3–5, poses in **base** frame using FK and the handout camera extrinsic `T_5_cam`), then **`solve_tower_of_hanoi()`** to move blocks from tower 0 → 2.  
- Uses Part 1 detection, Part 2 grasps, and robot control (e.g. URX) in the lab.

---

## Publishing to GitHub

1. Create a **new empty** repository on GitHub (no README/license there if you already committed locally).
2. In this project folder:

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

Use SSH instead of HTTPS if you prefer (`git@github.com:...`).

---

## License / course use

This code is for coursework. Follow your instructor’s collaboration and submission rules.
