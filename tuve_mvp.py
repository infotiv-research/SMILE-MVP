# ==========================================================
# TUVE MVP – FULL PyQt6 Port (Feature Complete + Info Panel)
# ==========================================================

import argparse
import sys, os, json, cv2, pickle, numpy as np, h5py, hdf5plugin  # noqa: F401
from functools import lru_cache
from pathlib import Path
from typing import Optional
from shapely.geometry import Point, Polygon

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# ==========================================================
# CONFIG (unchanged)
# ==========================================================

MASK_FOLDER = "Data/confidential_tuve_dataset/bev_res/"
DATASET_ROOT = "Data/confidential_tuve_dataset"
SAVE_PATH = "JSON-data/object_tracks.json"
WINDOW_RANGE = (500, 1700)
HORIZON = 10
BASE_SPEED = 10.0
VEHICLE_RADIUS = 20.0
MAX_DISTANCE = 150.0
MIN_COSINE = np.cos(np.deg2rad(60))
OOD_ROOT = "Data/ood_detections/"

# ==========================================================
# LOAD STATIC DATA
# ==========================================================

mask_files = sorted(
    f for f in os.listdir(MASK_FOLDER) if f.lower().endswith(".jpg")
)[WINDOW_RANGE[0] : WINDOW_RANGE[1]]

with open("Models/camera_visibility_lookup_table.pkl", "rb") as f:
    lookup_table = pickle.load(f)

with open(Path(DATASET_ROOT) / "dataset_config.json", "r") as f:
    dataset_config = json.load(f)

# ==========================================================
# DATA ADAPTORS (unchanged)
# ==========================================================

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

def get_visible_cameras(pos):
    ij = (int(pos[0]), int(pos[1]))
    return lookup_table.get(ij, {})

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


def parse_camera_list(spec):
    cameras = []
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_txt, end_txt = token.split("-", 1)
            start = int(start_txt)
            end = int(end_txt)
            step = 1 if end >= start else -1
            cameras.extend(str(cam) for cam in range(start, end + step, step))
        else:
            cameras.append(str(int(token)))

    ordered = []
    seen = set()
    for cam_id in cameras:
        if cam_id in seen:
            continue
        seen.add(cam_id)
        ordered.append(cam_id)
    return ordered


def squeeze_heatmap(arr):
    heatmap = np.asarray(arr, dtype=np.float32)
    while heatmap.ndim > 2 and heatmap.shape[0] == 1:
        heatmap = heatmap[0]
    if heatmap.ndim != 2:
        raise ValueError(f"Unexpected OOD heatmap shape: {heatmap.shape}")
    return heatmap


@lru_cache(maxsize=64)
def get_camera_frame_size(cam_id, frame_name):
    image_path = Path(DATASET_ROOT) / cam_id / frame_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read camera frame: {image_path}")
    h, w = image.shape[:2]
    return w, h


class OODHeatmapStore:
    def __init__(self, dataset_root, ood_root, config):
        self.dataset_root = Path(dataset_root)
        self.ood_root = Path(ood_root)
        self.config = config
        self.score_dtype = np.dtype([("id", np.uint64), ("ood_score", np.float32), ("pred", np.uint8)])
        self._score_cache = {}

    def frame_name(self, cam_id, idx):
        frames = self.config.get(str(cam_id), [])
        if idx < 0 or idx >= len(frames):
            return None
        return frames[idx]

    def frame_id(self, cam_id, idx):
        frame_name = self.frame_name(cam_id, idx)
        if not frame_name:
            return None
        return Path(frame_name).stem

    def frame_scores(self, cam_id):
        cam_id = str(cam_id)
        if cam_id not in self._score_cache:
            npz = self.ood_root / cam_id / "ood_score.npz"
            if not npz.exists():
                self._score_cache[cam_id] = {}
            else:
                arr = np.fromfile(npz, dtype=self.score_dtype)
                self._score_cache[cam_id] = {
                    str(int(item["id"])): (float(item["ood_score"]), bool(item["pred"]))
                    for item in arr
                }
        return self._score_cache[cam_id]

    @lru_cache(maxsize=128)
    def resized_heatmap(self, cam_id, frame_id, frame_name):
        if not frame_id or not frame_name:
            return None

        path = self.ood_root / str(cam_id) / "per_map.h5"
        if not path.exists():
            return None

        with h5py.File(path, "r") as handle:
            if frame_id not in handle:
                return None
            heatmap = squeeze_heatmap(handle[frame_id][()])

        width, height = get_camera_frame_size(str(cam_id), frame_name)
        return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)

    def sample(self, cam_id, idx, pixel_xy):
        frame_name = self.frame_name(cam_id, idx)
        frame_id = self.frame_id(cam_id, idx)
        heatmap = self.resized_heatmap(str(cam_id), frame_id, frame_name)
        if heatmap is None:
            return None

        x = int(np.clip(round(pixel_xy[0]), 0, heatmap.shape[1] - 1))
        y = int(np.clip(round(pixel_xy[1]), 0, heatmap.shape[0] - 1))
        frame_score = self.frame_scores(cam_id).get(frame_id)

        return {
            "frame_id": frame_id,
            "pixel": (x, y),
            "local_score": float(heatmap[y, x]),
            "frame_score": frame_score[0] if frame_score else None,
            "pred": frame_score[1] if frame_score else None,
        }

    @lru_cache(maxsize=128)
    def rgb_frame(self, cam_id, frame_name):
        if not frame_name:
            return None
        image_path = self.dataset_root / str(cam_id) / frame_name
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def overlay_frame(self, cam_id, idx, pixel_xy, alpha=0.45):
        frame_name = self.frame_name(cam_id, idx)
        frame_id = self.frame_id(cam_id, idx)
        heatmap = self.resized_heatmap(str(cam_id), frame_id, frame_name)
        rgb = self.rgb_frame(str(cam_id), frame_name)
        if heatmap is None or rgb is None:
            return None

        norm = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(norm.min())
        hi = float(norm.max())
        if hi > lo:
            norm = (norm - lo) / (hi - lo)
        else:
            norm = np.zeros_like(norm, dtype=np.float32)

        colored = (plt.get_cmap("plasma")(norm)[..., :3] * 255.0).astype(np.uint8)
        overlay = cv2.addWeighted(rgb, 1.0 - alpha, colored, alpha, 0.0)

        marker_xy = None
        if pixel_xy is not None:
            x = int(np.clip(round(pixel_xy[0]), 0, overlay.shape[1] - 1))
            y = int(np.clip(round(pixel_xy[1]), 0, overlay.shape[0] - 1))
            marker_xy = (x, y)
            cv2.drawMarker(
                overlay,
                marker_xy,
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )

        frame_score = self.frame_scores(cam_id).get(frame_id)
        return {
            "frame_id": frame_id,
            "frame_name": frame_name,
            "pixel": marker_xy,
            "frame_score": frame_score[0] if frame_score else None,
            "pred": frame_score[1] if frame_score else None,
            "image": overlay,
        }

# ==========================================================
# DATA LOAD
# ==========================================================

tracks, uncertainties, data_mode = load_tracks_auto(SAVE_PATH)

mask0 = cv2.imread(os.path.join(MASK_FOLDER, mask_files[0]), cv2.IMREAD_GRAYSCALE)
H, W = mask0.shape
NUM_FRAMES = min(len(tracks), len(mask_files))
ood_store = OODHeatmapStore(DATASET_ROOT, OOD_ROOT, dataset_config)

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
        self.alert_cam = []
        self.alert_ood = {}
        self.visible_camera_pixels = {}

    def update_frame(self, idx):
        self.frame_idx = idx
        self.alert_collision = False
        self.alert_cam = []
        self.alert_ood = {}
        self.visible_camera_pixels = {}

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
        visible_cameras = get_visible_cameras((vy, vx))
        self.alert_cam = sorted(map(str, visible_cameras.keys()))
        self.visible_camera_pixels = {
            str(cam_id): tuple(cam_xy) for cam_id, cam_xy in visible_cameras.items()
        }

        for f in range(idx, min(idx+HORIZON, len(tracks))):
            vp = Point(vehicle_position(f, W, data_mode, tracks))
            for box in tracks[f]:
                poly = get_bbox_polygon(box["center"], box["size"], box["angle"])
                if poly.buffer(VEHICLE_RADIUS).contains(vp):
                    self.alert_collision = True
                    break

        for cam_id, cam_xy in visible_cameras.items():
            sample = ood_store.sample(str(cam_id), idx, cam_xy)
            if sample is not None:
                self.alert_ood[str(cam_id)] = sample

        self.draw_idle()


class CameraOODViewer(FigureCanvasQTAgg):
    def __init__(self, camera_ids):
        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        super().__init__(self.fig)
        self.camera_ids = list(camera_ids)
        self.axes_by_cam = {}

        count = max(1, len(self.camera_ids))
        cols = min(3, count)
        rows = int(np.ceil(count / cols))
        for subplot_idx, cam_id in enumerate(self.camera_ids, start=1):
            ax = self.fig.add_subplot(rows, cols, subplot_idx)
            ax.axis("off")
            self.axes_by_cam[cam_id] = ax

    def update_views(self, idx, visible_camera_pixels, alert_ood):
        for cam_id in self.camera_ids:
            ax = self.axes_by_cam[cam_id]
            ax.clear()
            ax.axis("off")
            pixel_xy = visible_camera_pixels.get(cam_id)
            overlay = ood_store.overlay_frame(cam_id, idx, pixel_xy)

            if overlay is None:
                ax.set_title(f"Camera {cam_id}: overlay unavailable")
                continue

            if pixel_xy is None:
                black = np.zeros_like(overlay["image"])
                ax.imshow(black)
            else:
                ax.imshow(overlay["image"])
            frame_score = overlay["frame_score"]
            frame_score_txt = f"{frame_score:.3e}" if frame_score is not None else "-"
            pred_txt = int(overlay["pred"]) if overlay["pred"] is not None else "-"
            if cam_id in alert_ood:
                local_txt = f"{alert_ood[cam_id]['local_score']:.3f}"
                visibility_txt = "visible"
            else:
                local_txt = "-"
                visibility_txt = "not visible"
            ax.set_title(
                f"Cam {cam_id} | {visibility_txt}\n"
                f"local={local_txt} | frame={frame_score_txt} | pred={pred_txt}"
            )

        self.draw_idle()

# ==========================================================
# MAIN WINDOW + INFO PANEL
# ==========================================================

class MainWindow(QMainWindow):
    def __init__(self, show_ood_viewer=False, ood_cameras=None):
        super().__init__()
        self.setWindowTitle("TUVE MVP – Viewer")
        self.resize(1700 if show_ood_viewer else 1200, 900)

        self.viewer = MVPViewer()
        self.ood_cameras = list(ood_cameras or [])
        self.camera_viewer = CameraOODViewer(self.ood_cameras) if show_ood_viewer else None

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
        self.label_cam.setWordWrap(True)
        self.label_ood.setWordWrap(True)

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
        if self.camera_viewer is not None:
            main.addWidget(self.camera_viewer, 1)

        root = QWidget(); root.setLayout(main)
        self.setCentralWidget(root)

        self.play_btn.clicked.connect(self.toggle)
        self.reset_btn.clicked.connect(self.reset)
        self.slider.valueChanged.connect(self.seek)

        self.viewer.update_frame(0)
        self.refresh_status(0)

    def refresh_status(self, frame_idx):
        self.label_frame.setText(f"Frame: {frame_idx}")
        self.label_status.setText(
            "⚠ COLLISION" if self.viewer.alert_collision else "Status: OK"
        )
        self.label_cam.setText(
            "Camera: " + (", ".join(self.viewer.alert_cam) if self.viewer.alert_cam else "-")
        )
        if self.viewer.alert_ood:
            lines = []
            for cam_id in sorted(self.viewer.alert_ood):
                sample = self.viewer.alert_ood[cam_id]
                frame_score = sample["frame_score"]
                frame_score_txt = f"{frame_score:.3e}" if frame_score is not None else "-"
                pred_txt = int(sample["pred"]) if sample["pred"] is not None else "-"
                lines.append(
                    f"{cam_id}: local={sample['local_score']:.3f}, "
                    f"frame={frame_score_txt}, pred={pred_txt}"
                )
            self.label_ood.setText("OOD:\n" + "\n".join(lines))
        else:
            self.label_ood.setText("OOD: -")

        if self.camera_viewer is not None:
            self.camera_viewer.update_views(
                frame_idx,
                self.viewer.visible_camera_pixels,
                self.viewer.alert_ood,
            )

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

        self.refresh_status(i)

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
        self.slider.setValue(0)
        self.refresh_status(0)

    def seek(self, v):
        self.timer.stop()
        self.viewer.update_frame(v)
        self.refresh_status(v)

# ==========================================================
# ENTRY POINT
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TUVE MVP viewer")
    parser.add_argument(
        "--show-ood-viewer",
        action="store_true",
        help="Show camera RGB views with OOD heatmap overlays for currently visible cameras.",
    )
    parser.add_argument(
        "--ood-cameras",
        default="160-171",
        help=(
            "Comma-separated camera IDs and/or inclusive ranges for the OOD viewer, "
            "for example '160', '160,162,170', or '160-171'."
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    selected_ood_cameras = parse_camera_list(args.ood_cameras)
    if args.show_ood_viewer and not selected_ood_cameras:
        raise ValueError("No cameras selected for --show-ood-viewer. Check --ood-cameras.")
    app = QApplication(sys.argv)
    win = MainWindow(
        show_ood_viewer=args.show_ood_viewer,
        ood_cameras=selected_ood_cameras,
    )
    win.show()

    sys.exit(app.exec())
