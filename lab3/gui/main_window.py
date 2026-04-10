"""Main window: notebook tabs per lab part, status bar, Part 1 ArUco UI."""

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
from PIL import Image, ImageTk

from lab3 import __version__
from lab3.part1 import Part1PracticeRunner
from lab3.paths import (
    PRACTICE_IMAGE_NAME,
    default_practice_image_path,
    ensure_test_images_dir,
    list_test_images,
    repo_relative,
    test_images_dir,
)


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


# Preview pane: raw test image vs saved overlay (same stem under test_images/annotated/).
PREVIEW_ORIGINAL = "Original image"
PREVIEW_ANNOTATED = "Annotated overlay"


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

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Label(top, text="Test image:").pack(side=tk.LEFT)
        self._combo = ttk.Combobox(
            top,
            textvariable=self._image_choice,
            state="readonly",
            width=48,
        )
        self._combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self._combo.bind("<<ComboboxSelected>>", self._on_combo_changed)

        ttk.Button(top, text="Refresh list", command=self._refresh_list).pack(side=tk.LEFT)
        ttk.Button(top, text="Open test_images folder", command=self._open_test_images_folder).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        path_row = ttk.Frame(self)
        path_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(path_row, text="Path (repo-relative):").pack(side=tk.LEFT)
        ttk.Label(path_row, textvariable=self._path_label_var, foreground="gray").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(btn_row, text="Load preview", command=self._load_preview).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Run ArUco detection", command=self._run_detection_async).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(btn_row, text="Clear results", command=self._clear_results).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(btn_row, text=f"Use «{PRACTICE_IMAGE_NAME}»", command=self._select_handout_name).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        preview_frame = ttk.LabelFrame(mid, text="Preview")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        preview_hdr = ttk.Frame(preview_frame)
        preview_hdr.pack(fill=tk.X, padx=6, pady=(6, 0))
        ttk.Label(preview_hdr, text="Show:").pack(side=tk.LEFT)
        self._preview_mode_combo = ttk.Combobox(
            preview_hdr,
            textvariable=self._preview_mode,
            state="readonly",
            values=(PREVIEW_ORIGINAL, PREVIEW_ANNOTATED),
            width=22,
        )
        self._preview_mode_combo.pack(side=tk.LEFT, padx=(6, 0))
        self._preview_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_preview())

        self._preview_label = ttk.Label(preview_frame, anchor=tk.CENTER, text="No image loaded")
        self._preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        out_frame = ttk.LabelFrame(mid, text="Detection report")
        out_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._results = tk.Text(out_frame, height=18, width=52, wrap=tk.WORD, font=("Consolas", 9))
        sy = ttk.Scrollbar(out_frame, orient=tk.VERTICAL, command=self._results.yview)
        self._results.configure(yscrollcommand=sy.set)
        self._results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        sy.pack(side=tk.RIGHT, fill=tk.Y, pady=6, padx=(0, 6))

        hint = (
            "Add or remove images under the project «test_images» folder, then click «Refresh list». "
            "Use «Show» next to the preview to switch between the original and the saved overlay "
            "(test_images/annotated). Text reports go to test_images/reports/<name>_report.txt. "
            "After a successful run, the preview switches to the annotated image. "
            "Expected (handout): IDs 0 and 5; translations roughly "
            "(0.157, -0.100, 0.542) and (0.0865, -0.0836, 0.489); rotation ~π about y."
        )
        ttk.Label(self, text=hint, wraplength=900, justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(0, 8))

        self._sync_combo_from_disk(prefer_basename=self._initial_prefer_basename())
        self._load_preview()

    def _initial_prefer_basename(self) -> str | None:
        dp = default_practice_image_path()
        if dp.is_file():
            return dp.name
        return None

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
                f"(no images in {repo_relative(test_images_dir())} — add .png / .jpg and click Refresh list)"
            )

    def _on_combo_changed(self, *_args: object) -> None:
        self._update_path_label()
        self._load_preview()

    def _refresh_list(self) -> None:
        current = self._image_choice.get().strip()
        self._sync_combo_from_disk(prefer_basename=current or None)
        n = len(self._combo["values"])
        self._on_status(f"Found {n} image(s) in test_images.")
        self._load_preview()

    def _open_test_images_folder(self) -> None:
        d = ensure_test_images_dir()
        _open_folder(d)
        self._on_status(f"Opened: {repo_relative(d)}")

    def _select_handout_name(self) -> None:
        self._sync_combo_from_disk(prefer_basename=PRACTICE_IMAGE_NAME)
        self._on_status(f"Selection: {PRACTICE_IMAGE_NAME} (if present in test_images).")
        self._load_preview()

    def selected_path(self) -> Path | None:
        name = self._image_choice.get().strip()
        if not name:
            return None
        return self._paths_by_name.get(name)

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
                    image="",
                    text="No annotated file for this image yet.\nRun «Run ArUco detection» first.",
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
                "then click «Refresh list».",
                parent=self.winfo_toplevel(),
            )
            return

        self._on_busy(True)
        self._on_status("Running ArUco detection…")

        def work() -> None:
            markers, err, ann_path, report_path = self._runner.find_markers_in_file(
                path, save_annotated=True
            )

            def done() -> None:
                self._on_busy(False)
                if err is not None:
                    self._on_status(f"Error: {err}")
                    messagebox.showerror("Part 1", err, parent=self.winfo_toplevel())
                    return
                assert markers is not None
                report = self._runner.format_detection_report(
                    path, markers, annotated_path=ann_path, report_path=report_path
                )
                self._results.delete("1.0", tk.END)
                self._results.insert(tk.END, report)
                bits: list[str] = []
                if ann_path is not None:
                    bits.append(ann_path.name)
                if report_path is not None:
                    bits.append(report_path.name)
                extra = f"; saved {' + '.join(bits)}" if bits else ""
                self._on_status(f"Done — {len(markers)} marker(s) in {path.name}{extra}")
                if ann_path is not None:
                    self._preview_mode.set(PREVIEW_ANNOTATED)
                self._load_preview()

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def use_default_practice_image(self) -> None:
        """Menu hook: select the handout filename in the list when available."""
        self._select_handout_name()


class PlaceholderPanel(ttk.Frame):
    """Reserved tab for future Parts 2 / 3 features."""

    def __init__(self, master: tk.Misc, *, title: str, blurb: str) -> None:
        super().__init__(master)
        ttk.Label(self, text=title, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, padx=12, pady=(12, 4))
        ttk.Label(self, text=blurb, wraplength=720, justify=tk.LEFT).pack(
            anchor=tk.W, padx=12, pady=4
        )


class Lab3MainWindow(tk.Tk):
    """Root window: menu, notebook, status + progress."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"CDS/ME-235 Lab 3 — v{__version__}")
        self.minsize(960, 640)
        self.geometry("1024x700")

        self._build_menu()
        # Status bar must exist before the notebook: Part1Panel.__init__ calls set_status via _load_preview.
        self._build_status_bar()
        self._build_notebook()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_m = tk.Menu(menubar, tearoff=0)
        file_m.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_m)

        part1_m = tk.Menu(menubar, tearoff=0)
        part1_m.add_command(
            label="Select handout practice image in list",
            command=self._focus_part1_default_path,
        )
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
            "Part 1: test images in «test_images/».\n"
            "Parts 2–3: edit «lab3/part2.py» and «lab3/part3.py».",
            parent=self,
        )

    def _build_notebook(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        self._notebook = ttk.Notebook(outer)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._part1_panel = Part1Panel(
            self._notebook,
            on_status=self.set_status,
            on_busy=self.set_busy,
        )
        self._notebook.add(self._part1_panel, text="Part 1 — ArUco / vision")

        p2 = PlaceholderPanel(
            self._notebook,
            title="Part 2 — Parallel jaw grasping",
            blurb=(
                "Implement logic in «lab3/part2.py» (function get_grasp_pose). "
                "Add GUI controls here in «lab3/gui/main_window.py» when you are ready."
            ),
        )
        self._notebook.add(p2, text="Part 2 — Grasp")

        p3 = PlaceholderPanel(
            self._notebook,
            title="Part 3 — Tower of Hanoi",
            blurb=(
                "Implement «lab3/part3.py» (class HanoiSolver: marker_search, "
                "solve_tower_of_hanoi). Add robot/FK/IK imports from your earlier labs."
            ),
        )
        self._notebook.add(p3, text="Part 3 — Hanoi")

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_var = tk.StringVar(value="Ready.")
        self._status_label = ttk.Label(bar, textvariable=self._status_var, anchor=tk.W)
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=4)

        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self._progress.pack(side=tk.RIGHT, padx=8, pady=4)

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
