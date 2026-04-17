# ==========================================================
# TUVE MVP – FULL PyQt6 Port (Feature Complete + Info Panel)
# ==========================================================

import sys, os, json, cv2, pickle, numpy as np
from pathlib import Path
from typing import Tuple
from shapely.geometry import Point, Polygon

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ==========================================================
# CONFIG (unchanged)
# ==========================================================

MASK_FOLDER = "Data/(CONFIDENTIAL) Tuve dataset/bev_res/"
SAVE_PATH = "JSON-data/object_tracks.json"

def load_tracks_auto(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "object_tracks" in data:  # legacy
        return data["object_tracks"], data.get("uncertainty_boxes"), "legacy"

    if isinstance(data, list) and "object_list" in data[0]:
        return adapt_2026_format(data), None, "2026"

    raise ValueError("Unknown JSON format")

def adapt_2026_format(frames):
    tracks = []
    for frame in frames:
        boxes = []
        for obj in frame["object_list"]:
            poly = obj.get("associated_polygon", [])
            if len(poly) < 3:
                continue

            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]

            boxes.append({
                "center": [(min(xs)+max(xs))/2, (min(ys)+max(ys))/2],
                "size": [max(xs)-min(xs), max(ys)-min(ys)],
                "angle": obj.get("heading", {}).get("deg") or 0.0,
            })
        tracks.append(boxes)
    return tracks

# ==========================================================
# HELPERS (unchanged logic)
# ==========================================================

def getcamid(pos):
    ij = (int(pos[0]), int(pos[1]))
    return list(lookup_table.get(ij, {}).keys())

def normalize_cam_id(cam):
    if isinstance(cam, (list, tuple)):
        return str(cam[0]) if cam else None
    return str(cam) if cam else None

def load_ood(cam_id):
    npz = Path(OOD_ROOT) / cam_id / "ood_score.npz"
    if not npz.exists():
        return {}, []
    dtype = np.dtype([("id", np.uint64), ("ood_score", np.float32), ("pred", np.uint8)])
    arr = np.fromfile(npz, dtype=dtype)
    return {str(i):(float(s),bool(p)) for i,s,p in arr}, list(map(str,arr["id"]))

def get_bbox_polygon(center, size, angle):
    return Polygon(cv2.boxPoints(((center[0],center[1]), size, angle)))

def vehicle_position(idx, W, mode, anchor):
    if mode == "legacy":
        return (50 + BASE_SPEED*idx) % W, 230
    if anchor and idx < len(anchor) and anchor[idx]:
        return anchor[idx][0]["center"]
    return W*0.5, H*0.5

# ==========================================================
# DATA LOAD
# ==========================================================

tracks, uncertainties, data_mode = load_tracks_auto(SAVE_PATH)

mask0 = cv2.imread(os.path.join(MASK_FOLDER, mask_files[0]), cv2.IMREAD_GRAYSCALE)
H, W = mask0.shape
NUM_FRAMES = min(len(tracks), len(mask_files))

# ==========================================================
# VIEWER
# ==========================================================

class MVPViewer(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(8,8), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.img = self.ax.imshow(mask0, cmap="gray")
        self.vehicle_dot, = self.ax.plot([], [], "bo", markersize=8)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)

        self.box_artists = []
        self.frame_idx = 0

        # ★ exposed state for info panel
        self.alert_collision = False
        self.alert_cam = None
        self.alert_ood = None

    def update_frame(self, idx):
        self.frame_idx = idx
        self.alert_collision = False
        self.alert_cam = None
        self.alert_ood = None

        self.img.set_data(
            cv2.imread(os.path.join(MASK_FOLDER, mask_files[idx]), 0)
        )

        for a in self.box_artists:
            a.remove()
        self.box_artists.clear()

        for box in tracks[idx]:
            poly = get_bbox_polygon(box["center"], box["size"], box["angle"])
            (l,) = self.ax.plot(*poly.exterior.xy, "r-", lw=1)
            self.box_artists.append(l)

        vx, vy = vehicle_position(
            idx,
            W,
            mode=data_mode,
            anchor=tracks
        )
        self.vehicle_dot.set_data([vx], [vy])

        # -------- COLLISION LOGIC (UNCHANGED) --------
        pos = np.array([vx, vy])
        cam_id = normalize_cam_id(getcamid((vy, vx)))
        self.alert_cam = cam_id

        for f in range(idx, min(idx+HORIZON, len(tracks))):
            vp = Point(vehicle_position(f, W, data_mode, tracks))
            for box in tracks[f]:
                poly = get_bbox_polygon(box["center"], box["size"], box["angle"])
                if poly.buffer(VEHICLE_RADIUS).contains(vp):
                    self.alert_collision = True
                    break

        if cam_id:
            scores, keys = load_ood(cam_id)
            if idx < len(keys):
                self.alert_ood = scores.get(keys[idx])

        self.draw_idle()

# ==========================================================
# MAIN WINDOW + INFO PANEL
# ==========================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TUVE MVP – Viewer")
        self.resize(1200,900)

        self.viewer = MVPViewer()

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.next_frame)

        # --- Controls
        self.play_btn = QPushButton("▶ Play")
        self.reset_btn = QPushButton("🔄 Reset")
        self.load_btn = QPushButton("📂 Load")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, NUM_FRAMES-1)
        self.label_frame = QLabel("Frame: 0")

        # ★ Info panel
        self.label_status = QLabel("Status: OK")
        self.label_cam = QLabel("Camera: -")
        self.label_ood = QLabel("OOD: -")

        ctrl = QVBoxLayout()
        for w in (
            self.load_btn, self.play_btn, self.reset_btn,
            self.label_frame, self.slider,
            self.label_status, self.label_cam, self.label_ood
        ):
            ctrl.addWidget(w)

        main = QHBoxLayout()
        main.addWidget(self.viewer, 1)
        c = QWidget(); c.setLayout(ctrl); c.setFixedWidth(250)
        main.addWidget(c)

        root = QWidget(); root.setLayout(main)
        self.setCentralWidget(root)

        self.play_btn.clicked.connect(self.toggle)
        self.reset_btn.clicked.connect(self.reset)
        self.slider.valueChanged.connect(self.seek)

    def next_frame(self):
        # advance frame exactly once
        self.viewer.frame_idx = (self.viewer.frame_idx + 1) % NUM_FRAMES
        i = self.viewer.frame_idx

        # render
        self.viewer.update_frame(i)

        # update slider WITHOUT firing seek()
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)

        self.label_frame.setText(f"Frame: {i}")

        # update info panel
        self.label_status.setText(
            "⚠ COLLISION" if self.viewer.alert_collision else "Status: OK"
        )
        self.label_cam.setText(f"Camera: {self.viewer.alert_cam}")
        self.label_ood.setText(
            f"OOD: {self.viewer.alert_ood}" if self.viewer.alert_ood else "OOD: -"
        )

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.timer.start()
            self.play_btn.setText("⏸ Pause")

    def reset(self):
        self.timer.stop()
        self.viewer.update_frame(0)

    def seek(self, v):
        self.timer.stop()
        self.viewer.update_frame(v)

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    sys.exit(app.exec())
