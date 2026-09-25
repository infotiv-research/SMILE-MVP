# ==========================================================
# TUVE MVP
# ==========================================================

import argparse
import sys, os, json, cv2, pickle, numpy as np, h5py, hdf5plugin  # noqa: F401
from functools import lru_cache
from pathlib import Path
from shapely.geometry import Point, Polygon

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from undistorter import Undistorter


# ==========================================================
# CONFIG LOADER
# ==========================================================

def load_config(path="mvp-config.json"):
    with open(path, "r") as f:
        cfg = json.load(f)

    # Resolve paths relative to dataset_root
    root = Path(cfg["paths"]["dataset_root"])

    def p(key):
        return str(root / cfg["paths"][key])

    config = {
        "DATASET_ROOT": str(root),
        "MASK_FOLDER": p("mask_folder"),
        "DATASET_CONFIG_FILE": p("dataset_config_file"),
        "CAMERA_FOLDER": p("camera_folder"),
        "OOD_ROOT": p("ood_root"),
        "CALIB_INTR_ROOT": cfg["paths"]["calib_intr_root"],  # keep relative
        "CAMERA_LOOKUP_TABLE": p("camera_lookup_table"),
        "SAVE_PATH": p("save_path"),

        "WINDOW_RANGE": tuple(cfg["runtime"]["window_range"]),
        "HORIZON": cfg["runtime"]["horizon"],
        "BASE_SPEED": cfg["runtime"]["base_speed"],
        "VEHICLE_RADIUS": cfg["runtime"]["vehicle_radius"],
        "MAX_DISTANCE": cfg["runtime"]["max_distance"],
        "TRACK_STRIDE": cfg["runtime"].get("track_stride", 1),
        "MIN_COSINE": np.cos(np.deg2rad(cfg["runtime"]["min_cosine_deg"])),

        "UI": cfg["ui"],
        "CAMERA": cfg["camera"],
    }

    return config


class RuntimeContext:
    def __init__(self, config_path="mvp-config.json"):
        self.cfg = load_config(config_path)

        # ---- PATHS
        self.dataset_root = Path(self.cfg["DATASET_ROOT"])
        self.mask_folder = Path(self.cfg["MASK_FOLDER"])
        self.camera_folder = Path(self.cfg["CAMERA_FOLDER"])
        self.ood_root = Path(self.cfg["OOD_ROOT"])
        self.calib_intr_root = self.cfg["CALIB_INTR_ROOT"]

        # ---- RUNTIME PARAMS
        self.window_range = self.cfg["WINDOW_RANGE"]
        self.horizon = self.cfg["HORIZON"]
        self.base_speed = self.cfg["BASE_SPEED"]
        self.vehicle_radius = self.cfg["VEHICLE_RADIUS"]
        self.max_distance = self.cfg["MAX_DISTANCE"]
        self.track_stride = self.cfg.get("TRACK_STRIDE", 1)
        self.min_cosine = self.cfg["MIN_COSINE"]

        # ---- LAZY LOADED DATA (avoid IO during init)
        self._mask_files = None
        self._lookup_table = None
        self._dataset_config = None
        self._tracks_bundle = None
        self._mask0 = None

    # ------------ LAZY LOADERS (CACHED) ------------

    @property
    @lru_cache(maxsize=1)
    def mask_files(self):
        files = sorted(
            f for f in os.listdir(self.mask_folder)
            if f.lower().endswith(".jpg")
        )

        # --- apply window
        files = files[self.window_range[0]: self.window_range[1]]

        # pply stride
        stride = max(1, int(self.track_stride))
        if stride > 1:
            files = files[::stride]

        return files

    @property
    @lru_cache(maxsize=1)
    def lookup_table(self):
        with open(self.cfg["CAMERA_LOOKUP_TABLE"], "rb") as f:
            return pickle.load(f)

    @property
    @lru_cache(maxsize=1)
    def dataset_config(self):
        with open(self.cfg["DATASET_CONFIG_FILE"], "r") as f:
            return json.load(f)

    @property
    @lru_cache(maxsize=1)
    def tracks_bundle(self):
        return load_tracks_auto(self.cfg["SAVE_PATH"])

    @property
    @lru_cache(maxsize=1)
    def mask0(self):
        return cv2.imread(
            os.path.join(self.mask_folder, self.mask_files[0]),
            cv2.IMREAD_GRAYSCALE
        )

    # ------------ DERIVED ------------

    @property
    def tracks(self):
        return self.tracks_bundle[0]

    @property
    def uncertainties(self):
        return self.tracks_bundle[1]

    @property
    def data_mode(self):
        return self.tracks_bundle[2]

    @property
    def H(self):
        return self.mask0.shape[0]

    @property
    def W(self):
        return self.mask0.shape[1]

    @property
    def num_frames(self):
        return min(len(self.tracks), len(self.mask_files))


# ==========================================================
# DATA ADAPTORS 
# ==========================================================

def load_tracks_auto(path):
    import json

    with open(path, "r") as f:
        data = json.load(f)

    # --- LEGACY FORMAT ---
    if "object_tracks" in data:
        tracks = data["object_tracks"]
        u_boxes = data.get("uncertainty_boxes")

        # If no uncertainty boxes → create mock from bbox
        if u_boxes is None:
            u_boxes = []
            for frame in tracks:
                u_frame = []
                for box in frame:
                    u_frame.append({
                        "center": box["center"],
                        "size": box["size"],
                        "angle": box.get("angle", 0.0)
                    })
                u_boxes.append(u_frame)

        return tracks, u_boxes, "legacy"

    # --- 2026 FORMAT ---
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "object_list" in data[0]:
        tracks, u = adapt_2026_format(data)
        return tracks, u, "2026"

    raise ValueError("Unknown JSON format")

def adapt_2026_format(frames):
    from shapely.geometry import Polygon

    tracks = []
    uncs = []

    for frame_idx, frame in enumerate(frames):
        t_frame = []
        u_frame = []

        for obj in frame.get("object_list", []):
            poly = obj.get("associated_polygon") or []
            if len(poly) < 3:
                continue

            P = Polygon(poly)
            cx, cy = P.centroid.coords[0]

            xs, ys = zip(*poly)
            uw, uh = max(xs) - min(xs), max(ys) - min(ys)

            # heading (image-aligned)
            angle = (obj.get("heading") or {}).get("deg", 0.0)
            oid = obj.get("track_id", -1)

            # main bbox scaling from moving_avg and aligned to polygon
            bb = obj.get("bbox_moving_avg") or {}

            # Convert vehicle-based dimensions → scale factor
            base_w, base_h = uw, uh
            target_w = bb.get("width", base_w)
            target_h = bb.get("length", base_h)

            scale_w = target_w / base_w if base_w else 1.0
            scale_h = target_h / base_h if base_h else 1.0

            w = base_w * scale_w
            h = base_h * scale_h

            # --- MAIN BOX (aligned with polygon frame)
            t_frame.append({
                "frame": frame_idx,
                "object_id": oid,
                "center": [cx, cy],
                "size": [w, h],
                "angle": angle,  # same as polygon
            })

            # --- UNCERTAINTY (pure polygon bbox)
            u_frame.append({
                "frame": frame_idx,
                "object_id": oid,
                "center": [cx, cy],
                "size": [uw, uh],
                "angle": angle,  #  May need to fix here if we want axis-aligned uncertainty boxes instead of polygon-aligned
            })

        tracks.append(t_frame)
        uncs.append(u_frame)

    return tracks, uncs


# ==========================================================
# HELPERS 
# ==========================================================


def get_visible_cameras(pos, ctx):
    ij = (int(pos[0]), int(pos[1]))
    return ctx.lookup_table.get(ij, {})



def get_bbox_polygon(center, size, angle):
    return Polygon(cv2.boxPoints(((center[0],center[1]), size, angle)))


def vehicle_position(idx, ctx):
    speed = 2.0
    x = (idx * speed) % ctx.W   
    y = ctx.H * 0.5
    return x, y






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
def get_camera_frame_size(dataset_root, cam_id, frame_name, turbo=True):
    if turbo:
        return 640, 360

    image_path = Path(dataset_root) / cam_id / frame_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read camera frame: {image_path}")
    h, w = image.shape[:2]
    return w, h


@lru_cache(maxsize=32)
def get_undistorter(intr_root, cam_id, width, height):
    intr_root = Path(intr_root)

    candidates = [
        intr_root / f"{cam_id}_{width}x{height}.yaml",
        intr_root / f"{cam_id}.yaml",
    ]

    for path in candidates:
        if path.exists():
            return Undistorter(str(path), new_width=width, new_height=height)

    raise FileNotFoundError(
        f"No calibration file for camera {cam_id} at {intr_root}"
    )


@lru_cache(maxsize=128)
def load_mask(mask_folder, mask_files_tuple, idx):
    return cv2.imread(
        os.path.join(mask_folder, mask_files_tuple[idx]),
        cv2.IMREAD_GRAYSCALE
    )

def load_mask_ctx(ctx, idx):
    return load_mask(str(ctx.mask_folder), tuple(ctx.mask_files), idx)


def build_runtime_context(config_path="mvp-config.json"):
    return RuntimeContext(config_path)



class OODHeatmapStore:
    def __init__(self, dataset_root, ood_root, config):
        self.camera_root = Path(dataset_root)
        self.ood_root = Path(ood_root)
        self.config = config
        self.score_dtype = np.dtype([("id", np.uint64), ("ood_score", np.float32), ("pred", np.uint8)])
        self._score_cache = {}

    def undistorter(self, cam_id, frame_name):
        width, height = get_camera_frame_size(self.camera_root, str(cam_id), frame_name)
        return get_undistorter(self.config["CALIB_INTR_ROOT"], str(cam_id), width, height)


    def frame_info(self, cam_id, idx):
        cam_id = str(cam_id)

        frames = self.config["DATASET_CONFIG"].get(cam_id, [])

        if idx < 0 or idx >= len(frames):
            return None, None

        frame_name = frames[idx]
        if not frame_name:
            return None, None

        return frame_name, Path(frame_name).stem


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

    
    @staticmethod
    @lru_cache(maxsize=128)
    def _resized_heatmap_cached(ood_root, camera_root, intr_root, cam_id, frame_id, frame_name):
        if not frame_id or not frame_name:
            return None

        ood_root = Path(ood_root)
        camera_root = Path(camera_root)

        path = ood_root / str(cam_id) / "per_map.h5"
        if not path.exists():
            return None

        with h5py.File(path, "r") as handle:
            if frame_id not in handle:
                return None
            heatmap = squeeze_heatmap(handle[frame_id][()])

        # size
        width, height = get_camera_frame_size(camera_root, str(cam_id), frame_name)

        distorted = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)

        undistorter = get_undistorter(intr_root, str(cam_id), width, height)
        undistorted = undistorter.undistort(distorted.astype(np.float32))

        return undistorted.astype(np.float32)

    def resized_heatmap(self, cam_id, frame_id, frame_name):
        return self._resized_heatmap_cached(
            str(self.ood_root),
            str(self.camera_root),
            self.config["CALIB_INTR_ROOT"],
            str(cam_id),
            frame_id,
            frame_name,
        )


    def sample(self, cam_id, idx, pixel_xy):
        cam_id = str(cam_id)
        frame_name, frame_id = self.frame_info(cam_id, idx)

        heatmap = self.resized_heatmap(cam_id, frame_id, frame_name)
        if heatmap is None:
            return None

        h, w = heatmap.shape
        x = int(np.clip(round(pixel_xy[0]), 0, w - 1))
        y = int(np.clip(round(pixel_xy[1]), 0, h - 1))

        frame_score = self.frame_scores(cam_id).get(frame_id)

        return {
            "frame_id": frame_id,
            "pixel": (x, y),
            "local_score": float(heatmap[y, x]),
            "frame_score": frame_score[0] if frame_score else None,
            "pred": frame_score[1] if frame_score else None,
        }

    def rgb_frame(self, cam_id, frame_name):
        if not frame_name:
            return None
        image_path = self.camera_root / str(cam_id) / frame_name
        print("Loading image:", image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        undistorted = self.undistorter(cam_id, frame_name).undistort(bgr)
        return cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

    def overlay_frame(self, cam_id, idx, pixel_xy, alpha=0.45):
        cam_id = str(cam_id)
        frame_name, frame_id = self.frame_info(cam_id, idx)

        heatmap = self.resized_heatmap(cam_id, frame_id, frame_name)
        rgb = self.rgb_frame(cam_id, frame_name)
        if heatmap is None or rgb is None:
            return None

        norm = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = norm.min(), norm.max()

        norm = (norm - lo) / (hi - lo) if hi > lo else np.zeros_like(norm)

        colored = (plt.get_cmap("plasma")(norm)[..., :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(rgb, 1.0 - alpha, colored, alpha, 0.0)

        marker_xy = None
        if pixel_xy is not None:
            h, w = overlay.shape[:2]
            x = int(np.clip(round(pixel_xy[0]), 0, w - 1))
            y = int(np.clip(round(pixel_xy[1]), 0, h - 1))
            marker_xy = (x, y)

            cv2.drawMarker(
                overlay, marker_xy, (0, 255, 255),
                markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2
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
# VIEWER
# ==========================================================

class MVPViewer(FigureCanvasQTAgg):
    def __init__(self, ctx, ood_store):
        self.ctx = ctx
        self.ood_store = ood_store

        self.fig = Figure(figsize=(8, 8), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.img = self.ax.imshow(ctx.mask0, cmap="gray")
        self.vehicle_dot, = self.ax.plot([], [], "bo", markersize=8)

        self.ax.set_xlim(0, ctx.W)
        self.ax.set_ylim(ctx.H, 0)

        self.box_artists = []
        self.frame_idx = 0


    def update_frame(self, idx):
        ctx = self.ctx
        self.frame_idx = idx

        self.alert_collision = False
        self.alert_cam = []
        self.alert_ood = {}
        self.visible_camera_pixels = {}

        # --- background
        self.img.set_data(load_mask_ctx(ctx, idx))

        # --- clear previous boxes
        for a in self.box_artists:
            a.remove()
        self.box_artists.clear()

        # --- draw boxes
        tracks = ctx.tracks[idx]
        uncs = ctx.uncertainties[idx] if ctx.uncertainties else [None] * len(tracks)

        for box, u_box in zip(tracks, uncs):
            # --- main bbox (solid)
            poly = get_bbox_polygon(box["center"], box["size"], box["angle"])
            (l1,) = self.ax.plot(*poly.exterior.xy, "r-", lw=1)
            self.box_artists.append(l1)

            # --- uncertainty bbox (dashed, larger)
            if u_box:
                u_poly = get_bbox_polygon(u_box["center"], u_box["size"], u_box["angle"])
                (l2,) = self.ax.plot(*u_poly.exterior.xy, "r--", lw=1)
                self.box_artists.append(l2)

        # --- vehicle
        vx, vy = vehicle_position(idx, ctx)
        self.vehicle_dot.set_data([vx], [vy])

        # --- visible cameras
        visible_cameras = get_visible_cameras((vy, vx), ctx)
        self.alert_cam = sorted(map(str, visible_cameras.keys()))

        self.visible_camera_pixels = {
            str(cam_id): tuple(cam_xy)
            for cam_id, cam_xy in visible_cameras.items()
        }

        # --- collision detection
        horizon_end = min(idx + ctx.horizon, len(ctx.tracks))

        for f in range(idx, horizon_end):
            vp = Point(vehicle_position(f, ctx))

            for box in ctx.tracks[f]:
                poly = get_bbox_polygon(box["center"], box["size"], box["angle"])
                if poly.buffer(ctx.vehicle_radius).contains(vp):
                    self.alert_collision = True
                    break
            if self.alert_collision:
                break

        # --- OOD
        for cam_id, cam_xy in visible_cameras.items():
            sample = self.ood_store.sample(str(cam_id), idx, cam_xy)
            if sample is not None:
                self.alert_ood[str(cam_id)] = sample

        self.draw_idle()



class CameraOODViewer(FigureCanvasQTAgg):
    def __init__(self, camera_ids, ood_store):
        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        super().__init__(self.fig)
        self.camera_ids = list(camera_ids)
        self.axes_by_cam = {}
        self.ood_store = ood_store

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
            overlay = self.ood_store.overlay_frame(cam_id, idx, pixel_xy)

            if overlay is None:
                ax.set_title(f"cam{cam_id},ood:-", fontsize=9)
                continue

            if pixel_xy is None:
                black = np.zeros_like(overlay["image"])
                ax.imshow(black)
            else:
                ax.imshow(overlay["image"])
            pred_txt = int(overlay["pred"]) if overlay["pred"] is not None else "-"
            if cam_id in alert_ood:
                local_txt = f"{alert_ood[cam_id]['local_score']:.3f}"
            else:
                local_txt = "-"
            ax.set_title(f"cam{cam_id},ood:{local_txt}", fontsize=9)

        self.draw_idle()

# ==========================================================
# MAIN WINDOW + INFO PANEL
# ==========================================================

class MainWindow(QMainWindow):
    def __init__(self, ctx, show_ood_viewer=False, ood_cameras=None):
        super().__init__()
        self.ctx = ctx
        self.ood_store = OODHeatmapStore(
            ctx.camera_folder,
            ctx.ood_root,
            {
                "DATASET_CONFIG": ctx.dataset_config,
                "CALIB_INTR_ROOT": ctx.calib_intr_root,
            }
        )


        self.setWindowTitle("TUVE MVP – Viewer")
        self.resize(
            ctx.cfg["UI"]["wide_window_width"] if show_ood_viewer else ctx.cfg["UI"]["default_window_width"],
            ctx.cfg["UI"]["window_height"]
        )

        # --- viewers
        self.viewer = MVPViewer(ctx, self.ood_store)
        self.camera_viewer = (
            CameraOODViewer(ood_cameras, self.ood_store)
            if show_ood_viewer else None
        )

        # --- timer
        self.timer = QTimer(self)
        self.timer.setInterval(ctx.cfg["UI"]["timer_interval_ms"])
        self.timer.timeout.connect(self.next_frame)

        # --- controls
        self.play_btn = QPushButton("▶ Play")
        self.reset_btn = QPushButton("🔄 Reset")
        self.load_btn = QPushButton("📂 Load")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, ctx.num_frames - 1)

        self.label_frame = QLabel("Frame: 0")
        self.label_status = QLabel("Status: OK")
        self.label_cam = QLabel("Camera: -")
        self.label_ood = QLabel("OOD: -")

        self.label_cam.setWordWrap(True)
        self.label_ood.setWordWrap(True)

        # --- layout
        ctrl = QVBoxLayout()
        for w in (
            self.load_btn, self.play_btn, self.reset_btn,
            self.label_frame, self.slider,
            self.label_status, self.label_cam, self.label_ood
        ):
            ctrl.addWidget(w)

        main = QHBoxLayout()
        main.addWidget(self.viewer, 1)

        side = QWidget()
        side.setLayout(ctrl)
        side.setFixedWidth(250)
        main.addWidget(side)

        if self.camera_viewer:
            main.addWidget(self.camera_viewer, 1)

        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)

        # --- signals
        self.play_btn.clicked.connect(self.toggle)
        self.reset_btn.clicked.connect(self.reset)
        self.slider.valueChanged.connect(self.seek)

        # --- init
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
        self.viewer.frame_idx = (self.viewer.frame_idx + 1) % self.ctx.num_frames
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

    ctx = RuntimeContext()

    selected_ood_cameras = parse_camera_list(args.ood_cameras)

    if args.show_ood_viewer and not selected_ood_cameras:
        raise ValueError("No cameras selected")

    app = QApplication(sys.argv)

    win = MainWindow(
        ctx,
        show_ood_viewer=args.show_ood_viewer,
        ood_cameras=selected_ood_cameras,
    )

    win.show()
    sys.exit(app.exec())