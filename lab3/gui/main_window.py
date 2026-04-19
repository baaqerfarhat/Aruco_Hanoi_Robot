"""Main window — modern dark-themed notebook with tabs for each lab part."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import ttkbootstrap as ttb
    from ttkbootstrap.constants import (
        DANGER,
        DARK,
        INFO,
        OUTLINE,
        PRIMARY,
        SECONDARY,
        SUCCESS,
        WARNING,
    )

    _HAS_BOOTSTRAP = True
except ImportError:
    _HAS_BOOTSTRAP = False

from lab3 import __version__
from lab3.part1 import Part1PracticeRunner
from lab3.part2 import get_grasp_pose, calculate_gripper_percentage
from lab3.ur10e_kinematics import (
    UR10eKinematics,
    modified_joint_rad_to_classical_joint_deg,
    pose6_to_T_euler,
)
from lab3.paths import (
    PRACTICE_IMAGE_NAME,
    default_practice_image_path,
    ensure_test_images_dir,
    list_test_images,
    repo_relative,
    test_images_dir,
)

_MONO = ("Consolas", 9)
_HEADING = ("Segoe UI", 10, "bold")
_BODY = ("Segoe UI", 9)

_TEXT_DARK = dict(
    font=_MONO,
    bg="#212529",
    fg="#dee2e6",
    insertbackground="#dee2e6",
    selectbackground="#495057",
    selectforeground="#f8f9fa",
    relief=tk.FLAT,
    borderwidth=1,
    padx=8,
    pady=6,
)

PREVIEW_ORIGINAL = "Original image"
PREVIEW_ANNOTATED = "Annotated overlay"

PAD = dict(padx=12, pady=6)
PAD_SECTION = dict(padx=12, pady=(12, 4))


def _open_folder(path: Path) -> None:
    """Open a directory in the system file manager (best-effort)."""
    path = path.resolve()
    if not path.is_dir():
        return
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError:
        pass


def _make_text(parent, height=18, width=0, **extra) -> tk.Text:
    """Create a consistently styled monospaced Text widget."""
    kw = {**_TEXT_DARK, **extra}
    if width:
        kw["width"] = width
    t = tk.Text(parent, height=height, wrap=tk.WORD, **kw)
    return t


def _styled_btn(parent, text, command, style="", **kw):
    """Create a themed button. Use ``style`` for ttkbootstrap bootstyle strings."""
    if _HAS_BOOTSTRAP:
        return ttb.Button(parent, text=text, command=command, bootstyle=style, **kw)
    return ttk.Button(parent, text=text, command=command, **kw)


# ---------------------------------------------------------------------------
#  Part 1
# ---------------------------------------------------------------------------


class Part1Panel(ttk.Frame):
    """ArUco detection: test image dropdown, preview, run/clear, results text."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_status: Callable[[str], None],
        on_busy: Callable[[bool], None],
    ) -> None:
        super().__init__(master)
        self._on_status = on_status
        self._on_busy = on_busy
        self._runner = Part1PracticeRunner()
        self._photo: ImageTk.PhotoImage | None = None
        self._preview_mode = tk.StringVar(value=PREVIEW_ORIGINAL)
        self._image_choice = tk.StringVar(value="")
        self._paths_by_name: dict[str, Path] = {}
        self._path_label_var = tk.StringVar(value="")

        # --- image selector row ---
        top = ttk.Frame(self)
        top.pack(fill=tk.X, **PAD_SECTION)

        ttk.Label(top, text="Test image:", font=_HEADING).pack(side=tk.LEFT)
        self._combo = ttk.Combobox(
            top, textvariable=self._image_choice, state="readonly", width=44,
        )
        self._combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self._combo.bind("<<ComboboxSelected>>", self._on_combo_changed)

        _styled_btn(top, "Refresh", self._refresh_list, style=f"{INFO}-{OUTLINE}" if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        _styled_btn(top, "Open folder", self._open_test_images_folder, style=SECONDARY if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT
        )

        path_row = ttk.Frame(self)
        path_row.pack(fill=tk.X, padx=12, pady=(0, 4))
        ttk.Label(path_row, text="Path:", font=_BODY).pack(side=tk.LEFT)
        ttk.Label(path_row, textvariable=self._path_label_var, foreground="#6c757d", font=_BODY).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        # --- action buttons ---
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=12, pady=4)

        _styled_btn(btn_row, "Load preview", self._load_preview, style=INFO if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT
        )
        _styled_btn(btn_row, "Run ArUco detection", self._run_detection_async, style=SUCCESS if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        _styled_btn(btn_row, "Clear results", self._clear_results, style=SECONDARY if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        _styled_btn(
            btn_row, f"Use \u00ab{PRACTICE_IMAGE_NAME}\u00bb", self._select_handout_name,
            style=f"{WARNING}-{OUTLINE}" if _HAS_BOOTSTRAP else "",
        ).pack(side=tk.LEFT, padx=(8, 0))

        # --- preview + report (side by side) ---
        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        preview_frame = ttk.LabelFrame(mid, text="Preview")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        preview_hdr = ttk.Frame(preview_frame)
        preview_hdr.pack(fill=tk.X, padx=8, pady=(6, 0))
        ttk.Label(preview_hdr, text="Show:").pack(side=tk.LEFT)
        self._preview_mode_combo = ttk.Combobox(
            preview_hdr, textvariable=self._preview_mode, state="readonly",
            values=(PREVIEW_ORIGINAL, PREVIEW_ANNOTATED), width=22,
        )
        self._preview_mode_combo.pack(side=tk.LEFT, padx=(6, 0))
        self._preview_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_preview())

        self._preview_label = ttk.Label(preview_frame, anchor=tk.CENTER, text="No image loaded")
        self._preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        out_frame = ttk.LabelFrame(mid, text="Detection report")
        out_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._results = _make_text(out_frame, height=18, width=52)
        sy = ttk.Scrollbar(out_frame, orient=tk.VERTICAL, command=self._results.yview)
        self._results.configure(yscrollcommand=sy.set)
        self._results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        sy.pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=(0, 8))

        hint = (
            "Add images under \u00abtest_images\u00bb, click Refresh. "
            "Expected (handout): IDs 0, 5 | t \u2248 (0.157, -0.100, 0.542) & (0.087, -0.084, 0.489) | "
            "rotation \u2248\u03c0 about y."
        )
        ttk.Label(self, text=hint, wraplength=1000, justify=tk.LEFT, foreground="#6c757d").pack(
            anchor=tk.W, padx=12, pady=(0, 8)
        )

        self._sync_combo_from_disk(prefer_basename=self._initial_prefer_basename())
        self._load_preview()

    # ---- helpers ----

    def _initial_prefer_basename(self) -> str | None:
        dp = default_practice_image_path()
        return dp.name if dp.is_file() else None

    def _sync_combo_from_disk(self, *, prefer_basename: str | None = None) -> None:
        ensure_test_images_dir()
        paths = list_test_images()
        self._paths_by_name = {p.name: p for p in paths}
        names = [p.name for p in paths]
        self._combo.configure(values=names)
        chosen = ""
        if prefer_basename and prefer_basename in self._paths_by_name:
            chosen = prefer_basename
        elif PRACTICE_IMAGE_NAME in self._paths_by_name:
            chosen = PRACTICE_IMAGE_NAME
        elif names:
            chosen = names[0]
        self._image_choice.set(chosen)
        self._update_path_label()

    def _update_path_label(self) -> None:
        name = self._image_choice.get().strip()
        if name and name in self._paths_by_name:
            self._path_label_var.set(repo_relative(self._paths_by_name[name]))
        else:
            self._path_label_var.set(
                f"(no images in {repo_relative(test_images_dir())} \u2014 add .png/.jpg and click Refresh)"
            )

    def _on_combo_changed(self, *_args: object) -> None:
        self._update_path_label()
        self._load_preview()

    def _refresh_list(self) -> None:
        current = self._image_choice.get().strip()
        self._sync_combo_from_disk(prefer_basename=current or None)
        self._on_status(f"Found {len(self._combo['values'])} image(s) in test_images.")
        self._load_preview()

    def _open_test_images_folder(self) -> None:
        d = ensure_test_images_dir()
        _open_folder(d)
        self._on_status(f"Opened: {repo_relative(d)}")

    def _select_handout_name(self) -> None:
        self._sync_combo_from_disk(prefer_basename=PRACTICE_IMAGE_NAME)
        self._on_status(f"Selection: {PRACTICE_IMAGE_NAME}")
        self._load_preview()

    def selected_path(self) -> Path | None:
        name = self._image_choice.get().strip()
        return self._paths_by_name.get(name) if name else None

    def _load_preview(self) -> None:
        path = self.selected_path()
        if path is None or not path.is_file():
            self._preview_label.configure(image="", text="No image selected" if path is None else "File missing")
            self._photo = None
            return
        mode = self._preview_mode.get()
        load_path = path
        if mode == PREVIEW_ANNOTATED:
            load_path = Part1PracticeRunner.annotated_output_path(path)
            if not load_path.is_file():
                self._preview_label.configure(
                    image="", text="No annotated file yet.\nRun detection first.",
                )
                self._photo = None
                self._on_status(f"No overlay at {load_path.name}")
                return
        bgr = cv2.imread(str(load_path), cv2.IMREAD_COLOR)
        if bgr is None:
            self._on_status(f"Could not decode: {load_path}")
            self._preview_label.configure(image="", text="Could not read image")
            self._photo = None
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((520, 420), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self._preview_label.configure(image=self._photo, text="")
        label = "Annotated" if mode == PREVIEW_ANNOTATED else "Original"
        self._on_status(f"Preview ({label}): {load_path.name}")

    def _clear_results(self) -> None:
        self._results.delete("1.0", tk.END)
        self._on_status("Results cleared.")

    def _run_detection_async(self) -> None:
        path = self.selected_path()
        if path is None or not path.is_file():
            messagebox.showwarning(
                "Part 1",
                f"Choose an image in the dropdown or add files under:\n{repo_relative(test_images_dir())}\n"
                "then click Refresh.",
                parent=self.winfo_toplevel(),
            )
            return
        self._on_busy(True)
        self._on_status("Running ArUco detection\u2026")

        def work() -> None:
            markers, err, ann_path, report_path = self._runner.find_markers_in_file(path, save_annotated=True)

            def done() -> None:
                self._on_busy(False)
                if err is not None:
                    self._on_status(f"Error: {err}")
                    messagebox.showerror("Part 1", err, parent=self.winfo_toplevel())
                    return
                assert markers is not None
                report = self._runner.format_detection_report(
                    path, markers, annotated_path=ann_path, report_path=report_path,
                )
                self._results.delete("1.0", tk.END)
                self._results.insert(tk.END, report)
                bits: list[str] = []
                if ann_path is not None:
                    bits.append(ann_path.name)
                if report_path is not None:
                    bits.append(report_path.name)
                extra = f"; saved {' + '.join(bits)}" if bits else ""
                self._on_status(f"Done \u2014 {len(markers)} marker(s) in {path.name}{extra}")
                if ann_path is not None:
                    self._preview_mode.set(PREVIEW_ANNOTATED)
                self._load_preview()

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def use_default_practice_image(self) -> None:
        self._select_handout_name()


# ---------------------------------------------------------------------------
#  Part 2
# ---------------------------------------------------------------------------


class Part2Panel(ttk.Frame):
    """Part 2 \u2014 compute grasp pose from detected ArUco tags."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_status: Callable[[str], None],
        on_busy: Callable[[bool], None],
    ) -> None:
        super().__init__(master)
        self._on_status = on_status
        self._on_busy = on_busy
        self._runner = Part1PracticeRunner()

        # --- configuration form ---
        cfg = ttk.LabelFrame(self, text="Configuration")
        cfg.pack(fill=tk.X, **PAD_SECTION)

        inner = ttk.Frame(cfg)
        inner.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(inner, text="Camera pose  (x, y, z, roll, pitch, yaw):", font=_BODY).grid(
            row=0, column=0, sticky=tk.W
        )
        self._cam_pose_var = tk.StringVar(value="0.0, -0.4, 0.4, 3.14159, 0, 3.14159")
        ttk.Entry(inner, textvariable=self._cam_pose_var, width=50, font=_MONO).grid(
            row=0, column=1, padx=(8, 0), sticky=tk.W
        )

        ttk.Label(inner, text="Prism width [m]:", font=_BODY).grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self._width_var = tk.StringVar(value="0.08")
        ttk.Entry(inner, textvariable=self._width_var, width=12, font=_MONO).grid(
            row=1, column=1, padx=(8, 0), sticky=tk.W, pady=(6, 0)
        )

        ttk.Label(inner, text="Test image:", font=_BODY).grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        self._image_choice = tk.StringVar(value="")
        self._paths_by_name: dict[str, Path] = {}
        self._combo = ttk.Combobox(inner, textvariable=self._image_choice, state="readonly", width=48)
        self._combo.grid(row=2, column=1, padx=(8, 0), sticky=tk.W, pady=(6, 0))
        self._sync_combo()

        # --- buttons ---
        btn = ttk.Frame(self)
        btn.pack(fill=tk.X, padx=12, pady=6)
        _styled_btn(btn, "Compute grasp pose", self._run_async, style=SUCCESS if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT
        )
        _styled_btn(btn, "Clear", self._clear, style=SECONDARY if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        _styled_btn(btn, "Refresh images", self._sync_combo, style=f"{INFO}-{OUTLINE}" if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # --- results ---
        res_frame = ttk.LabelFrame(self, text="Results")
        res_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self._results = _make_text(res_frame, height=22)
        sy = ttk.Scrollbar(res_frame, orient=tk.VERTICAL, command=self._results.yview)
        self._results.configure(yscrollcommand=sy.set)
        self._results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        sy.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)

    def _sync_combo(self) -> None:
        ensure_test_images_dir()
        paths = list_test_images()
        self._paths_by_name = {p.name: p for p in paths}
        names = [p.name for p in paths]
        self._combo.configure(values=names)
        if not self._image_choice.get() and names:
            prefer = PRACTICE_IMAGE_NAME if PRACTICE_IMAGE_NAME in self._paths_by_name else names[0]
            self._image_choice.set(prefer)

    def _clear(self) -> None:
        self._results.delete("1.0", tk.END)
        self._on_status("Part 2 results cleared.")

    def _run_async(self) -> None:
        name = self._image_choice.get().strip()
        img_path = self._paths_by_name.get(name) if name else None
        if img_path is None or not img_path.is_file():
            messagebox.showwarning("Part 2", "Select a valid test image first.", parent=self.winfo_toplevel())
            return
        try:
            cam_vals = [float(x.strip()) for x in self._cam_pose_var.get().split(",")]
            if len(cam_vals) != 6:
                raise ValueError
            cam_pose = np.array(cam_vals, dtype=np.float64)
        except (ValueError, TypeError):
            messagebox.showerror("Part 2", "Camera pose must be 6 comma-separated numbers.", parent=self.winfo_toplevel())
            return
        try:
            prism_w = float(self._width_var.get().strip())
        except (ValueError, TypeError):
            messagebox.showerror("Part 2", "Prism width must be a number.", parent=self.winfo_toplevel())
            return

        self._on_busy(True)
        self._on_status("Part 2: detecting markers and computing grasp pose\u2026")

        def work() -> None:
            markers, err, _, _ = self._runner.find_markers_in_file(img_path, save_annotated=False)
            T_base_cam = pose6_to_T_euler(cam_pose)
            lines: list[str] = []
            lines.append(f"Image: {img_path.name}")
            lines.append(f"Camera pose (RPY): {cam_vals}")
            lines.append(f"Prism width: {prism_w} m")
            lines.append(f"Gripper close %: {calculate_gripper_percentage(prism_w):.1f}%")
            lines.append("")
            if err is not None:
                lines.append(f"ERROR: {err}")
            elif markers is None or len(markers) == 0:
                lines.append("No markers detected.")
            else:
                lines.append(f"Detected {len(markers)} marker(s):\n")
                for m in markers:
                    T_base_marker = T_base_cam @ m.T_cam_marker
                    T_bg = get_grasp_pose(T_base_marker, prism_w)
                    lines.append(f"--- Marker ID {m.marker_id} ---")
                    lines.append(f"T_cam_marker:\n{m.T_cam_marker}\n")
                    lines.append(f"T_base_marker:\n{T_base_marker}\n")
                    lines.append(f"T_base_gripper (grasp pose):\n{T_bg}\n")
                    lines.append(f"Gripper position: {np.round(T_bg[:3, 3], 5)}")
                    lines.append(f"Gripper z-axis:   {np.round(T_bg[:3, 2], 5)}\n")
            text = "\n".join(lines)

            def done() -> None:
                self._on_busy(False)
                self._results.delete("1.0", tk.END)
                self._results.insert(tk.END, text)
                self._on_status(f"Part 2 done \u2014 {len(markers or [])} marker(s) processed.")

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------------
#  Part 3
# ---------------------------------------------------------------------------


class Part3Panel(ttk.Frame):
    """Part 3 \u2014 Tower of Hanoi with live camera, robot, and status tracking."""

    _FEED_INTERVAL_MS = 60

    class _StopRequested(Exception):
        pass

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_status: Callable[[str], None],
        on_busy: Callable[[bool], None],
    ) -> None:
        super().__init__(master)
        self._on_status = on_status
        self._on_busy = on_busy
        self._solver = None
        self._solver_thread: threading.Thread | None = None
        self._feed_running = False
        self._worker_active = False
        self._stop_requested = False
        self._cam_photo: ImageTk.PhotoImage | None = None

        # --- connection panel ---
        conn = ttk.LabelFrame(self, text="Robot Connection")
        conn.pack(fill=tk.X, **PAD_SECTION)

        r0 = ttk.Frame(conn)
        r0.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(r0, text="Robot IP:", font=_BODY).pack(side=tk.LEFT)
        self._ip_var = tk.StringVar(value="192.168.0.2")
        ttk.Entry(r0, textvariable=self._ip_var, width=18, font=_MONO).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(r0, text="Camera source:", font=_BODY).pack(side=tk.LEFT)
        self._cam_var = tk.StringVar(value="0")
        ttk.Entry(r0, textvariable=self._cam_var, width=6, font=_MONO).pack(side=tk.LEFT, padx=(6, 14))

        self._use_robot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r0, text="Connect to real robot + gripper", variable=self._use_robot_var).pack(side=tk.LEFT)

        # --- action buttons ---
        btn = ttk.Frame(self)
        btn.pack(fill=tk.X, padx=12, pady=6)
        _styled_btn(btn, "Initialize solver", self._init_solver, style=PRIMARY if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT
        )
        _styled_btn(btn, "Run marker search", self._run_search_async, style=INFO if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        _styled_btn(btn, "Solve Tower of Hanoi", self._run_hanoi_async, style=SUCCESS if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        sep = ttk.Separator(btn, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 6), pady=2)

        _styled_btn(btn, "Stop Robot", self._stop_robot, style=f"{WARNING}" if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )
        _styled_btn(btn, "STOP ALL", self._emergency_stop, style=DANGER if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        sep2 = ttk.Separator(btn, orient=tk.VERTICAL)
        sep2.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 6), pady=2)

        _styled_btn(btn, "Clear log", self._clear, style=SECONDARY if _HAS_BOOTSTRAP else "").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # --- process status ---
        status_frame = ttk.LabelFrame(self, text="Process Status")
        status_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        sf = ttk.Frame(status_frame)
        sf.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(sf, text="STATUS", font=("Segoe UI", 8, "bold"), foreground="#6c757d").grid(
            row=0, column=0, sticky=tk.W
        )
        self._proc_status_var = tk.StringVar(value="Idle")
        self._status_lbl = ttk.Label(sf, textvariable=self._proc_status_var, font=("Segoe UI", 11, "bold"),
                                     foreground="#00bc8c")
        self._status_lbl.grid(row=1, column=0, sticky=tk.W, padx=(0, 30))

        ttk.Label(sf, text="CURRENT STEP", font=("Segoe UI", 8, "bold"), foreground="#6c757d").grid(
            row=0, column=1, sticky=tk.W
        )
        self._proc_step_var = tk.StringVar(value="\u2014")
        ttk.Label(sf, textvariable=self._proc_step_var, font=("Segoe UI", 10)).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 30)
        )

        ttk.Label(sf, text="MARKERS", font=("Segoe UI", 8, "bold"), foreground="#6c757d").grid(
            row=0, column=2, sticky=tk.W
        )
        self._proc_markers_var = tk.StringVar(value="0")
        ttk.Label(sf, textvariable=self._proc_markers_var, font=("Segoe UI", 11, "bold"),
                  foreground="#3498db").grid(row=1, column=2, sticky=tk.W, padx=(0, 30))

        ttk.Label(sf, text="MOVES", font=("Segoe UI", 8, "bold"), foreground="#6c757d").grid(
            row=0, column=3, sticky=tk.W
        )
        self._proc_moves_var = tk.StringVar(value="0 / 7")
        ttk.Label(sf, textvariable=self._proc_moves_var, font=("Segoe UI", 11, "bold"),
                  foreground="#e74c3c").grid(row=1, column=3, sticky=tk.W)

        # --- camera feed + log ---
        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        cam_frame = ttk.LabelFrame(mid, text="Camera Feed")
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))
        cam_inner = ttk.Frame(cam_frame)
        cam_inner.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._cam_label = ttk.Label(cam_inner, anchor=tk.CENTER, text="No camera",
                                    font=("Segoe UI", 9), foreground="#6c757d")
        self._cam_label.pack(fill=tk.BOTH, expand=True)
        self._cam_label.configure(width=48)

        log_frame = ttk.LabelFrame(mid, text="Log")
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._log = _make_text(log_frame, height=16)
        sy = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log.yview)
        self._log.configure(yscrollcommand=sy.set)
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        sy.pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=(0, 8))

    # --- helpers ---

    def _set_process(self, status: str, step: str = "\u2014",
                     markers: str | None = None, moves: str | None = None) -> None:
        self._proc_status_var.set(status)
        self._proc_step_var.set(step)
        if markers is not None:
            self._proc_markers_var.set(markers)
        if moves is not None:
            self._proc_moves_var.set(moves)
        color_map = {
            "Idle": "#6c757d", "Ready": "#00bc8c", "Initializing...": "#f39c12",
            "Marker Search": "#3498db", "Solving Hanoi": "#e74c3c",
            "Done": "#00bc8c", "Error": "#e74c3c", "STOPPED": "#e74c3c",
        }
        self._status_lbl.configure(foreground=color_map.get(status, "#dee2e6"))

    def _set_process_safe(self, status: str, step: str = "\u2014",
                          markers: str | None = None, moves: str | None = None) -> None:
        self.after(0, lambda: self._set_process(status, step, markers, moves))

    def _append(self, text: str) -> None:
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)

    def _clear(self) -> None:
        self._log.delete("1.0", tk.END)
        self._on_status("Part 3 log cleared.")

    # --- camera feed ---

    def _start_feed(self) -> None:
        if self._feed_running:
            return
        self._feed_running = True
        self._poll_camera()

    def _stop_feed(self) -> None:
        self._feed_running = False

    def _poll_camera(self) -> None:
        if not self._feed_running:
            return
        if self._solver is not None and self._solver._cap is not None and not self._worker_active:
            ok, frame = self._solver._cap.read()
            if ok and frame is not None:
                self._show_frame(frame)
        self.after(self._FEED_INTERVAL_MS, self._poll_camera)

    def _show_frame(self, bgr_frame) -> None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((440, 340), Image.Resampling.LANCZOS)
        self._cam_photo = ImageTk.PhotoImage(pil)
        self._cam_label.configure(image=self._cam_photo, text="")

    def _show_frame_safe(self, bgr_frame) -> None:
        frame_copy = bgr_frame.copy()
        self.after(0, lambda: self._show_frame(frame_copy))

    # --- solver init ---

    def _init_solver(self) -> None:
        from lab3.part3 import HanoiSolver, UR_IP

        robot = None
        gripper = None
        cam_src: int | str | None = None
        try:
            raw = self._cam_var.get().strip()
            cam_src = int(raw) if raw else None
        except ValueError:
            cam_src = raw if raw else None
        ip = self._ip_var.get().strip() or UR_IP

        self._set_process("Initializing...", "Connecting")

        if self._use_robot_var.get():
            try:
                import urx
                robot = urx.Robot(ip)
                robot.set_tcp((0, 0, 0, 0, 0, 0))
                robot.set_payload(1.5, (0, 0, 0.07))
                self._append(f"Connected to robot at {ip}")
            except Exception as exc:
                self._append(f"Robot connection failed: {exc}")
                self._set_process("Error", "Robot connect failed")
                messagebox.showerror("Part 3", f"Could not connect to robot:\n{exc}", parent=self.winfo_toplevel())
                return
            try:
                from pyrobotiqur import RobotiqGripper
                gripper = RobotiqGripper(ip, port=63352, timeout=20.0)
                gripper.connect()
                gripper.activate()
                self._append("Gripper connected and activated.")
            except Exception as exc:
                self._append(f"Gripper connection failed: {exc}")
                self._set_process("Error", "Gripper connect failed")
                messagebox.showerror("Part 3", f"Could not connect to gripper:\n{exc}", parent=self.winfo_toplevel())
                return
        else:
            self._append("Dry-run mode (no robot / gripper connection).")

        try:
            if self._solver is not None:
                self._stop_feed()
                self._solver.release_camera()
            self._solver = HanoiSolver(robot=robot, gripper=gripper, camera_source=cam_src)
            self._append(f"HanoiSolver initialised (camera={cam_src}).")
            self._set_process("Ready", "Solver initialised", "0", "0 / 7")
            self._on_status("Part 3: solver ready.")
            self._start_feed()
        except Exception as exc:
            self._append(f"Solver init failed: {exc}")
            self._set_process("Error", str(exc))
            messagebox.showerror("Part 3", f"Solver init error:\n{exc}", parent=self.winfo_toplevel())

    # --- marker search ---

    def _run_search_async(self) -> None:
        if self._solver is None:
            messagebox.showwarning("Part 3", "Initialise the solver first.", parent=self.winfo_toplevel())
            return
        self._stop_requested = False
        self._on_busy(True)
        self._worker_active = True
        self._on_status("Part 3: running marker search\u2026")
        self._set_process("Marker Search", "Starting\u2026", "0")
        self._append("\n\u2500\u2500\u2500 Marker Search \u2500\u2500\u2500")

        solver = self._solver
        original_capture = solver.capture_frame
        panel = self
        grid_count = [0]

        def patched_capture():
            if panel._stop_requested:
                raise Part3Panel._StopRequested()
            grid_count[0] += 1
            panel._set_process_safe("Marker Search", f"Viewpoint {grid_count[0]}")
            frame = original_capture()
            if frame is not None:
                panel._show_frame_safe(frame)
            return frame

        def work() -> None:
            solver.capture_frame = patched_capture
            try:
                markers = solver.marker_search()
                lines: list[str] = []
                if not markers:
                    lines.append("No markers found.")
                else:
                    lines.append(f"Found {len(markers)} marker(s):")
                    for mid, T in sorted(markers.items()):
                        pos = np.round(T[:3, 3], 4)
                        lines.append(f"  ID {mid}: position {pos}")
                        lines.append(f"  T_base_marker:\n{T}\n")
                text = "\n".join(lines)
                n_markers = str(len(markers))
            except Exception as exc:
                text = f"marker_search error: {exc}"
                n_markers = "?"
            finally:
                solver.capture_frame = original_capture

            def done() -> None:
                self._worker_active = False
                self._on_busy(False)
                self._append(text)
                self._set_process("Ready", "Search complete", n_markers)
                self._on_status("Part 3: marker search finished.")

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    # --- solve hanoi ---

    def _run_hanoi_async(self) -> None:
        if self._solver is None:
            messagebox.showwarning("Part 3", "Initialise the solver first.", parent=self.winfo_toplevel())
            return
        self._stop_requested = False
        self._on_busy(True)
        self._worker_active = True
        self._on_status("Part 3: solving Tower of Hanoi\u2026")
        self._set_process("Solving Hanoi", "Starting\u2026", moves="0 / 7")
        self._append("\n\u2500\u2500\u2500 Tower of Hanoi \u2500\u2500\u2500")

        original_print = __builtins__["print"] if isinstance(__builtins__, dict) else getattr(__builtins__, "print")

        solver = self._solver
        original_capture = solver.capture_frame
        panel = self
        move_count = [0]

        def patched_capture():
            if panel._stop_requested:
                raise Part3Panel._StopRequested()
            frame = original_capture()
            if frame is not None:
                panel._show_frame_safe(frame)
            return frame

        def captured_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            if msg.startswith("Move block"):
                move_count[0] += 1
                panel._set_process_safe("Solving Hanoi", msg, moves=f"{move_count[0]} / 7")
            elif "Retrying" in msg:
                panel._set_process_safe("Solving Hanoi", f"Re-searching\u2026", moves=f"{move_count[0]} / 7")
            panel.after(0, lambda: panel._append(msg))
            original_print(*args, **kwargs)

        def work() -> None:
            import builtins
            builtins.print = captured_print
            solver.capture_frame = patched_capture
            try:
                solver.solve_tower_of_hanoi()
                result_msg = "Tower of Hanoi complete!"
            except NotImplementedError as exc:
                result_msg = f"NotImplementedError: {exc}"
            except Exception as exc:
                result_msg = f"Error: {exc}"
            finally:
                builtins.print = original_print
                solver.capture_frame = original_capture

            final = result_msg

            def done() -> None:
                self._worker_active = False
                self._on_busy(False)
                self._append(f"\n{final}")
                ok = "complete" in final.lower()
                self._set_process("Done" if ok else "Error", final,
                                  moves=f"{move_count[0]} / 7")
                self._on_status(f"Part 3: {final}")

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    # --- stop controls ---

    def _stop_robot(self) -> None:
        # Halt robot movement immediately (doesn't kill the running task)
        if self._solver is None or self._solver._robot is None:
            self._append("No robot connected.")
            return
        try:
            self._solver._robot.stopj(2.0)
            self._append("Robot stopped (stopj).")
            self._on_status("Part 3: robot halted.")
        except Exception as exc:
            self._append(f"Stop robot failed: {exc}")

    def _emergency_stop(self) -> None:
        # Kill everything: abort worker thread, stop robot, release camera
        self._stop_requested = True
        self._append("\n>>> STOP ALL <<<")

        # Stop robot movement
        if self._solver is not None and self._solver._robot is not None:
            try:
                self._solver._robot.stopj(2.0)
                self._append("Robot stopped.")
            except Exception as exc:
                self._append(f"Stop robot error: {exc}")

        # Open gripper to release anything held
        if self._solver is not None and self._solver._gripper is not None:
            try:
                self._solver._gripper.move_percent(0)
                self._append("Gripper opened.")
            except Exception as exc:
                self._append(f"Open gripper error: {exc}")

        # Stop camera feed and release
        self._stop_feed()
        if self._solver is not None:
            self._solver.release_camera()
            self._append("Camera released.")

        self._cam_label.configure(image="", text="No camera")
        self._cam_photo = None
        self._worker_active = False
        self._on_busy(False)
        self._set_process("STOPPED", "Emergency stop", "0", "0 / 7")
        self._on_status("Part 3: EMERGENCY STOP.")


# ---------------------------------------------------------------------------
#  Main window
# ---------------------------------------------------------------------------


class Lab3MainWindow(ttb.Window if _HAS_BOOTSTRAP else tk.Tk):  # type: ignore[misc]
    """Root window with modern dark theme, notebook tabs, and status bar."""

    def __init__(self) -> None:
        if _HAS_BOOTSTRAP:
            super().__init__(
                title=f"CDS/ME-235 Lab 3 \u2014 v{__version__}",
                themename="darkly",
                minsize=(1100, 720),
                size=(1280, 820),
            )
        else:
            super().__init__()
            self.title(f"CDS/ME-235 Lab 3 \u2014 v{__version__}")
            self.minsize(1100, 720)
            self.geometry("1280x820")

        self._build_menu()
        self._build_status_bar()
        self._build_notebook()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_m = tk.Menu(menubar, tearoff=0)
        file_m.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_m)

        part1_m = tk.Menu(menubar, tearoff=0)
        part1_m.add_command(label="Select handout practice image", command=self._focus_part1_default_path)
        menubar.add_cascade(label="Part 1", menu=part1_m)

        help_m = tk.Menu(menubar, tearoff=0)
        help_m.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_m)
        self.config(menu=menubar)

    def _focus_part1_default_path(self) -> None:
        self._notebook.select(0)
        self._part1_panel.use_default_practice_image()

    def _about(self) -> None:
        messagebox.showinfo(
            "About",
            f"CDS/ME-235 Lab 3\nVersion {__version__}\n\n"
            "Part 1 \u2014 ArUco tag detection & pose estimation\n"
            "Part 2 \u2014 Parallel-jaw grasp pose computation\n"
            "Part 3 \u2014 Tower of Hanoi with robot + camera",
            parent=self,
        )

    def _build_notebook(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        self._notebook = ttk.Notebook(outer)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        self._part1_panel = Part1Panel(self._notebook, on_status=self.set_status, on_busy=self.set_busy)
        self._notebook.add(self._part1_panel, text="  Part 1 \u2014 ArUco / Vision  ")

        self._part2_panel = Part2Panel(self._notebook, on_status=self.set_status, on_busy=self.set_busy)
        self._notebook.add(self._part2_panel, text="  Part 2 \u2014 Grasp Pose  ")

        self._part3_panel = Part3Panel(self._notebook, on_status=self.set_status, on_busy=self.set_busy)
        self._notebook.add(self._part3_panel, text="  Part 3 \u2014 Tower of Hanoi  ")

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Separator(bar, orient=tk.HORIZONTAL).pack(fill=tk.X)

        inner = ttk.Frame(bar)
        inner.pack(fill=tk.X, padx=12, pady=6)

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(inner, textvariable=self._status_var, anchor=tk.W, font=_BODY).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self._progress = ttk.Progressbar(inner, mode="indeterminate", length=160)
        self._progress.pack(side=tk.RIGHT, padx=(12, 0))

    def set_status(self, message: str) -> None:
        self._status_var.set(message)

    def set_busy(self, busy: bool) -> None:
        if busy:
            self._progress.start(12)
        else:
            self._progress.stop()


def main() -> None:
    app = Lab3MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
