"""Dedicated, read-only GUI for generated Scenario 1/2 replay artifacts.

This module is launched only by ``tuve_mvp_2026_04_16.py --scenario`` so the
ordinary MVP viewer and its layout remain unchanged.
"""

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon


ROOT = Path(__file__).resolve().parent
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return json.load(source)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config():
    with (ROOT / "mvp-config.json").open() as source:
        config = json.load(source)
    required = ("scenario_1_instances", "scenario_2_instances",
                "intersection_area", "robot_operating_area")
    missing = [key for key in required if key not in config.get("scenario_paths", {})]
    if missing:
        raise KeyError("mvp-config.json scenario_paths is missing: {}".format(
            ", ".join(missing)))
    return config


def scenario_files(config):
    paths = config["scenario_paths"]
    return {
        "Scenario 1 — Intersection": resolve_path(paths["scenario_1_instances"]),
        "Scenario 2 — Outside intersection": resolve_path(paths["scenario_2_instances"]),
    }


def load_polygon(path):
    if not path.exists():
        return []
    return read_json(path).get("polygon_image_px", [])


def configured_bev_folders(config):
    """Return BEV directories from the normal MVP dataset configuration."""
    paths = config["paths"]
    dataset_root = resolve_path(paths["dataset_root"])
    folders = [dataset_root / paths["mask_folder"]]
    extra_mask = config.get("extra_paths", {}).get("mask_folder")
    if extra_mask:
        folders.append(dataset_root / extra_mask)
    return folders


def find_bev_images(config):
    """Index numeric and camera-prefixed BEV image filenames by timestamp."""
    images = {}
    for folder in configured_bev_folders(config):
        if not folder.is_dir():
            continue
        for image in folder.glob("*.jpg"):
            match = re.search(r"(\d+)$", image.stem)
            if match is None:
                continue
            images.setdefault(int(match.group(1)), image)
        if images:
            break
    return images


class ReplayCanvas(FigureCanvasQTAgg):
    def __init__(self, intersection, operating_area, images):
        self.figure = Figure(figsize=(10, 7), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.intersection = intersection
        self.operating_area = operating_area
        self.images = images
        self.last_image = None

    @staticmethod
    def cached_position(instance, index):
        if "_live_position" in instance:
            return instance["_live_position"]
        cached = instance.get("precomputed_behavior", {})
        positions = cached.get("positions", [])
        if positions:
            return positions[min(index, len(positions) - 1)]
        path = instance.get("robot_path", [])
        return path[-1] if path else [0.0, 0.0]

    @staticmethod
    def cached_action(instance, index):
        if "_live_action" in instance:
            return instance["_live_action"]
        actions = instance.get("precomputed_behavior", {}).get("actions", [])
        return actions[min(index, len(actions) - 1)] if actions else "N/A"

    def render_replay(self, instances, index):
        self.ax.clear()
        base = max(instances, key=lambda item: len(item.get("frames", [])))
        frames = base.get("frames", [])
        frame = frames[min(index, len(frames) - 1)] if frames else {}
        timestamp = int(frame.get("timestamp", 0))
        image_path = self.images.get(timestamp)
        if image_path:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                self.ax.imshow(image, cmap="gray", origin="upper")
                height, width = image.shape[:2]
            else:
                height, width = 600, 1500
        else:
            height, width = 600, 1500
            self.ax.imshow(np.zeros((height, width)), cmap="gray", origin="upper")

        if self.operating_area:
            self.ax.add_patch(MplPolygon(self.operating_area, closed=True, facecolor="limegreen",
                              edgecolor="lime", alpha=0.14, linewidth=2, zorder=10))
        if self.intersection:
            self.ax.add_patch(MplPolygon(self.intersection, closed=True, facecolor="deepskyblue",
                              edgecolor="cyan", alpha=0.20, linewidth=2, zorder=11))
        self.ax.text(0.02, 0.035, "ROBOT OPERATING AREA", transform=self.ax.transAxes,
                     color="lime", weight="bold", fontsize=8, va="bottom", zorder=30)
        self.ax.text(0.02, 0.075, "INTERSECTION", transform=self.ax.transAxes,
                     color="cyan", weight="bold", fontsize=8, va="bottom", zorder=30)

        focus_id = base.get("focus_track_id")
        for obj in frame.get("objects", []):
            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue
            focus = int(obj.get("track_id", -1)) == int(focus_id)
            self.ax.add_patch(MplPolygon(polygon, closed=True, fill=False,
                              edgecolor="red" if focus else "orange", linewidth=2 if focus else 1,
                              zorder=20))
            center = np.mean(np.asarray(polygon, dtype=float), axis=0)
            self.ax.text(center[0], center[1], str(obj.get("track_id")), color="yellow",
                         fontsize=7, zorder=21)

        for color_index, instance in enumerate(instances):
            path = instance.get("robot_path", [])
            if len(path) >= 2:
                x, y = zip(*path)
                color = COLORS[color_index % len(COLORS)]
                self.ax.plot(x, y, "--", color=color, linewidth=1.6, alpha=0.75, zorder=15)
                px, py = self.cached_position(instance, index)
                self.ax.scatter([px], [py], s=65, color=color, edgecolors="white",
                                linewidths=0.5, zorder=25)

        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Scenario replay — recorded forklifts; precomputed AMR behavior")
        self.draw_idle()


class ScenarioReplayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TUVE MVP — Scenario Replay")
        self.resize(1450, 900)
        self.config = load_config()
        scenario_paths = self.config["scenario_paths"]
        self.intersection = load_polygon(resolve_path(scenario_paths["intersection_area"]))
        self.operating_area = load_polygon(resolve_path(scenario_paths["robot_operating_area"]))
        self.canvas = ReplayCanvas(self.intersection, self.operating_area,
                                   find_bev_images(self.config))
        self.payloads = {label: read_json(path) for label, path in scenario_files(self.config).items()
                         if path.exists()}
        if not self.payloads:
            raise FileNotFoundError(
                "No scenario replay artifacts found at paths configured in mvp-config.json"
            )
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.advance)
        self.frame_index = 0
        self.use_precomputed = "--precomputed" in sys.argv
        self.live_replays = {}
        self.live_mvp = None
        self.live_app = None

        self.scenario_select = QComboBox()
        self.scenario_select.addItems(self.payloads.keys())
        self.window_select = QComboBox()
        self.path_select = QComboBox()
        self.load_button = QPushButton("Load selection")
        self.play_button = QPushButton("▶ Play")
        self.reset_button = QPushButton("↻ Reset")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel()
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumWidth(250)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Scenario")); controls.addWidget(self.scenario_select)
        controls.addWidget(QLabel("Recorded time window")); controls.addWidget(self.window_select)
        controls.addWidget(QLabel("Robot instance")); controls.addWidget(self.path_select)
        controls.addWidget(self.load_button)
        controls.addWidget(self.play_button); controls.addWidget(self.reset_button)
        controls.addWidget(self.frame_label); controls.addWidget(self.slider)
        controls.addWidget(QLabel("Replay details")); controls.addWidget(self.info_label)
        controls.addStretch()
        panel = QWidget(); panel.setLayout(controls); panel.setFixedWidth(280)
        layout = QHBoxLayout(); layout.addWidget(self.canvas, 1); layout.addWidget(panel)
        root = QWidget(); root.setLayout(layout); self.setCentralWidget(root)

        self.scenario_select.currentTextChanged.connect(self.populate_windows)
        self.window_select.currentTextChanged.connect(self.populate_paths)
        self.load_button.clicked.connect(self.load_selection)
        self.play_button.clicked.connect(self.toggle_play)
        self.reset_button.clicked.connect(self.reset)
        self.slider.valueChanged.connect(self.seek)
        self.populate_windows()

    def instances_for_window(self):
        window_id = self.window_select.currentData()
        return [item for item in self.payloads[self.scenario_select.currentText()]["instances"]
                if item.get("window_id", item.get("id")) == window_id]

    def selected_instances(self):
        variants = self.instances_for_window()
        selected = self.path_select.currentData()
        return variants if selected == "__all__" else [item for item in variants if item["id"] == selected]

    def populate_windows(self):
        payload = self.payloads[self.scenario_select.currentText()]
        windows = {}
        for item in payload["instances"]:
            windows.setdefault(item.get("window_id", item["id"]), item)
        self.window_select.blockSignals(True); self.window_select.clear()
        for window_id, item in windows.items():
            self.window_select.addItem("{} — forklift {}".format(window_id, item.get("focus_track_id", "?")), window_id)
        self.window_select.blockSignals(False)
        self.populate_paths()

    def populate_paths(self):
        variants = self.instances_for_window()
        self.path_select.blockSignals(True); self.path_select.clear()
        self.path_select.addItem("All robot instances (parallel)", "__all__")
        for item in variants:
            self.path_select.addItem(item.get("robot_path_label", item["id"]), item["id"])
        self.path_select.blockSignals(False)
        self.load_selection()

    def load_selection(self):
        self.timer.stop(); self.play_button.setText("▶ Play"); self.frame_index = 0
        self.live_replays = {}
        instances = self.selected_instances()
        maximum = max(len(item.get("frames", [])) for item in instances) - 1
        self.slider.blockSignals(True); self.slider.setRange(0, max(0, maximum)); self.slider.setValue(0); self.slider.blockSignals(False)
        self.render()

    def render(self):
        instances = self.selected_instances()
        if not self.use_precomputed:
            self.render_live_behavior(instances)
        self.canvas.render_replay(instances, self.frame_index)
        actions = Counter(ReplayCanvas.cached_action(item, self.frame_index) for item in instances)
        first = instances[0]
        self.frame_label.setText("Frame {} / {}".format(self.frame_index, self.slider.maximum()))
        self.info_label.setText(
            "Focus forklift: {}\nExpected coverage: {}\nActions: {}\n\n{} behavior; robots do not interact with each other.".format(
                first.get("focus_track_id", "?"), first.get("coverage", "recorded replay"),
                ", ".join("{}={}".format(key, value) for key, value in sorted(actions.items())),
                "Precomputed" if self.use_precomputed else "Live existing-MVP",
            )
        )

    def render_live_behavior(self, instances):
        if self.live_mvp is None:
            from precompute_scenario_behaviors import ExistingMvpReplay, load_existing_mvp
            self.live_mvp = load_existing_mvp()
            self.live_app = QApplication.instance()
            self._live_replay_type = ExistingMvpReplay
        for instance in instances:
            key = instance["id"]
            replay = self.live_replays.get(key)
            if replay is None:
                replay = self._live_replay_type(self.live_mvp, self.live_app, instance)
                self.live_replays[key] = replay
            position, action = replay.advance_to(self.frame_index)
            instance["_live_position"] = position
            instance["_live_action"] = action

    def seek(self, value):
        self.frame_index = value; self.render()

    def advance(self):
        if self.frame_index >= self.slider.maximum():
            self.timer.stop(); self.play_button.setText("↻ Replay"); return
        self.slider.setValue(self.frame_index + 1)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop(); self.play_button.setText("▶ Play")
        else:
            if self.frame_index >= self.slider.maximum(): self.reset()
            self.timer.start(); self.play_button.setText("⏸ Pause")

    def reset(self):
        self.timer.stop(); self.play_button.setText("▶ Play"); self.slider.setValue(0)


def run():
    app = QApplication(sys.argv)
    window = ScenarioReplayWindow()
    window.show()
    return app.exec()
