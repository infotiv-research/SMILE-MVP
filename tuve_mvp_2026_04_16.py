# ==========================================================
# TUVE MVP FULL PyQt6 Port (Feature Complete + Info Panel)
# ==========================================================

import sys, os, json, cv2, pickle 
import numpy as np
import math
from pathlib import Path
from typing import Tuple
from shapely.geometry import Point, Polygon
import joblib

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel,
    QFileDialog, QComboBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

# ==========================================================
# CONFIG (unchanged)
# ==========================================================

# Paths and runtime values come from mvp-config.json
# (run this script from the repository root)
CONFIG_FILE = "mvp-config.json"

with open(CONFIG_FILE, "r") as _f:
    CFG = json.load(_f)

DATA_ROOT = Path(CFG["paths"]["dataset_root"])
EXTRA_PATHS = CFG["extra_paths"]

MASK_FOLDER = str(DATA_ROOT / EXTRA_PATHS["mask_folder"])
SAVE_PATH = str(DATA_ROOT / CFG["paths"]["save_path"])
PRED_PATH = str(DATA_ROOT / EXTRA_PATHS["predictions"])
OOD_ROOT = str(DATA_ROOT / CFG["paths"]["ood_root"])
CAMERA_LOOKUP_PATH = str(DATA_ROOT / CFG["paths"]["camera_lookup_table"])
OCCUPANCY_PATH = str(DATA_ROOT / EXTRA_PATHS["occupancy_projection"])
PATH_DB = str(DATA_ROOT / EXTRA_PATHS["robot_paths"])

VEHICLE_RADIUS = CFG["runtime"]["vehicle_radius"]
MAX_DISTANCE = CFG["runtime"]["max_distance"]

OOD_CACHE = {}

# ==========================================================
# LOAD STATIC DATA
# ==========================================================

mask_lookup = {}

for f in os.listdir(MASK_FOLDER):
    if not f.lower().endswith(".jpg"):
        continue

    try:
        ts = int(Path(f).stem)
        mask_lookup[ts] = f
    except ValueError:
        pass

with open(CAMERA_LOOKUP_PATH, "rb") as f:
    lookup_table = pickle.load(f)

# ==========================================================
# LOAD OCCUPANCY MAP
# ==========================================================
with open(OCCUPANCY_PATH, "rb") as f:
    occ_proj = pickle.load(f)

P = np.asarray(occ_proj["P"], dtype=np.float64)
K = np.asarray(occ_proj["K"], dtype=np.float64)
R = np.asarray(occ_proj["R"], dtype=np.float64)
t = np.asarray(occ_proj["t"], dtype=np.float64).reshape(3, 1)

# ==========================================================
# DATA ADAPTORS 
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

        timestamp = frame.get("time_stamp")

        frame_objects = []

        for obj in frame.get("object_list", []):

            track_id = obj.get("track_id")

            pos = obj.get("position_3d", {})
            vel = obj.get("velocity_3d", {})
            bbox = obj.get("bbox_moving_avg", {})
            heading = obj.get("heading", {})

            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))

            vx = float(vel.get("x", 0.0))
            vy = float(vel.get("y", 0.0))

            width = float(bbox.get("width", 0.0))
            length = float(bbox.get("length", 0.0))

            angle = heading.get("deg")

            if angle is None:
                angle = heading_from_velocity(vx, vy)

            frame_objects.append({
                "track_id": track_id,
                "timestamp": timestamp,

                "center_world": [x, y],

                "velocity_world": [vx, vy],

                # NOTE:
                # width/length are occupancy pixel dimensions,
                # not world dimensions
                "size_px": [
                    width,
                    length
                ],

                "polygon": obj.get(
                    "associated_polygon",
                    []
                ),

                "angle": float(angle)
            })

        tracks.append({
            "timestamp": timestamp,
            "objects": frame_objects
        })

    return tracks

# ==========================================================
# HELPERS (unchanged logic)
# ==========================================================

def heading_from_velocity(vx, vy):

    speed = math.hypot(vx, vy)

    if speed < 1e-6:
        return 0.0

    return math.degrees(
        math.atan2(vy, vx)
    )

def getcamid(pos):
    ij = (int(pos[0]), int(pos[1]))
    return list(lookup_table.get(ij, {}).keys())

def normalize_cam_id(cam):
    if isinstance(cam, (list, tuple)):
        return str(cam[0]) if cam else None
    return str(cam) if cam else None

def load_ood(cam_id):

    if cam_id in OOD_CACHE:
        return OOD_CACHE[cam_id]

    npz = (
        Path(OOD_ROOT)
        / cam_id
        / "ood_score.npz"
    )

    if not npz.exists():
        OOD_CACHE[cam_id] = ({}, [])
        return OOD_CACHE[cam_id]

    dtype = np.dtype([
        ("id", np.uint64),
        ("ood_score", np.float32),
        ("pred", np.uint8)
    ])

    arr = np.fromfile(npz, dtype=dtype)

    result = (
        {
            str(i):
            (float(s), bool(p))
            for i, s, p in arr
        },
        list(map(str, arr["id"]))
    )

    OOD_CACHE[cam_id] = result

    return result



# ==========================================================
# DATA LOAD
# ==========================================================

tracks, uncertainties, data_mode = load_tracks_auto(SAVE_PATH)

first_ts = tracks[0]["timestamp"]
mask_file = mask_lookup.get(first_ts)

if mask_file is None:
    raise RuntimeError(
        f"No BEV image for timestamp {first_ts}"
    )
mask0 = cv2.imread(
    os.path.join(MASK_FOLDER, mask_file),
    cv2.IMREAD_GRAYSCALE
)
H, W = mask0.shape
NUM_FRAMES = len(tracks)


# ==========================================================
# LOAD PRECOMPUTED ROUTE + MANEUVER PREDICTIONS
# ==========================================================

with open(PRED_PATH, "r") as f:
    pred_data = json.load(f)

prediction_lookup = {}

for frame in pred_data:

    ts = frame.get("time_stamp")

    per_track = {}

    for pred in frame.get(
        "predictions",
        []
    ):

        per_track[
            pred["track_id"]
        ] = pred

    prediction_lookup[ts] = per_track

# ==========================================================
# PRECOMPUTED TRACK HISTORY
# ==========================================================

TRACK_HISTORY = {}

for frame_idx, frame in enumerate(tracks):

    for obj in frame["objects"]:

        tid = obj["track_id"]

        pts = np.asarray(
            obj["polygon"]
        )

        if len(pts) < 3:
            continue

        cx = float(
            np.mean(pts[:, 0])
        )

        cy = float(
            np.mean(pts[:, 1])
        )

        TRACK_HISTORY.setdefault(
            tid,
            []
        ).append(
            (
                frame_idx,
                cx,
                cy
            )
        )

# ==========================================================
# TRACK UNCERTAINTY HISTORY
# ==========================================================

TRACK_UNCERTAINTY = {}

for frame in tracks:

    for obj in frame["objects"]:

        tid = obj["track_id"]

        poly = np.asarray(
            obj["polygon"],
            dtype=np.float32
        )

        if len(poly) < 3:
            continue

        center = np.mean(poly, axis=0)

        area = cv2.contourArea(
            poly.astype(np.float32)
        )

        TRACK_UNCERTAINTY.setdefault(
            tid,
            []
        ).append({

            "center":
                center,

            "area":
                area,

            "frame":
                frame["timestamp"]
        })

# ==========================================================
# AUTO BUILD ENTRY POINTS FROM ROUTE PREDICTIONS
# ==========================================================

ENTRY_SAMPLES = {}

for timestamp, pred_frame in prediction_lookup.items():

    for tid, pred in pred_frame.items():

        route = pred.get("route")

        if not route:
            continue

        entry_id = route.get("from")

        if entry_id is None:
            continue

        hist = TRACK_HISTORY.get(
            int(tid),
            []
        )

        if not hist:
            continue

        first_pt = hist[0]

        ENTRY_SAMPLES.setdefault(
            entry_id,
            []
        ).append(
            (
                first_pt[1],
                first_pt[2]
            )
        )

ENTRY_POINTS = {}

for entry_id, pts in ENTRY_SAMPLES.items():

    pts = np.asarray(pts)

    ENTRY_POINTS[entry_id] = (
        float(np.mean(pts[:, 0])),
        float(np.mean(pts[:, 1]))
    )

# ==========================================================
# ROBOT PATH LIBRARY
# ==========================================================

class RobotPath:

    def __init__(self, name):
        self.name = name
        self.speed = 2.0
        self.waypoints = []
        self.start_frame = 0
        self.end_frame = None
        self.associated_track = None

DEFAULT_PATHS = {

    "Path_1": RobotPath("Path_1")
}


def save_paths():
    data = {}
    for name, path in DEFAULT_PATHS.items():
        data[name] = {
            "speed":
                path.speed,
            "start_frame":
                path.start_frame,
            "end_frame":
                path.end_frame,
            "associated_track":
                path.associated_track,
            "waypoints":
                path.waypoints
        }
    with open(PATH_DB, "w") as f:
        json.dump(
            data,
            f,
            indent=2
        )

def load_paths():
    if not os.path.exists(PATH_DB):
        return
    with open(PATH_DB, "r") as f:
        data = json.load(f)
    DEFAULT_PATHS.clear()
    for name, p in data.items():
        path = RobotPath(name)
        path.speed = p.get(
            "speed",
            2.0
        )
        path.start_frame = p.get(
            "start_frame",
            0
        )
        path.end_frame = p.get(
            "end_frame",
            None
        )
        path.associated_track = p.get(
            "associated_track",
            None
        )
        path.waypoints = [
            tuple(x)
            for x in p.get(
                "waypoints",
                []
            )
        ]
        DEFAULT_PATHS[name] = path

load_paths()

if len(DEFAULT_PATHS) == 0:

    DEFAULT_PATHS["Path_1"] = RobotPath(
        "Path_1"
    )
# ==========================================================
# VIEWER
# ==========================================================

class MVPViewer(FigureCanvasQTAgg):

    def __init__(self):

        self.fig = Figure(figsize=(8, 8), tight_layout=True)
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)

        self.img = self.ax.imshow(mask0, cmap="gray")

        self.vehicle_dot, = self.ax.plot(
            [], [],
            "bo",
            markersize=10
        )

        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)

        self.box_artists = []
        self.entry_markers = []
        self.trajectory_artist = None
        self.track_history = {}        

        self.frame_idx = 0

        self.alert_collision = False
        self.alert_cam = None
        self.alert_ood = None
        self.active_track_id = None

        self.route_info = None
        self.mode_info = None

        self.track_ids_current = []
        self.entry_artists = []
        self.trajectory_artist = None        
        self.current_traj_len = 0
        self.minimum_ttc = 999.0
        self.current_trust = 1.0
        self.runtime_state = "NORMAL"
        self.decision_action = "GO"
        self.near_miss = False
        self.failed_test = False
        self.evidence_log = []

        # ------------------------
        # AGENT STATE
        # ------------------------

        self.current_path_name = "Path_1"

        if len(DEFAULT_PATHS["Path_1"].waypoints) == 0:
            DEFAULT_PATHS["Path_1"].waypoints = [(100,240),(700,240)]

        self.agent_x = 0.0
        self.agent_y = 0.0

        self.agent_start_x = 0.0
        self.agent_start_y = 0.0

        self.agent_speed_factor = 1.0
        self.current_waypoint_idx = 1

        self.snap_agent_to_active_path()

        self.path_artist = None

        self.path_edit_mode = False

        self.dragging = False

        self.mpl_connect(
            "button_press_event",
            self.on_press
        )

        self.mpl_connect(
            "button_release_event",
            self.on_release
        )

        self.mpl_connect(
            "motion_notify_event",
            self.on_motion
        )

    def log_evidence(self):

        self.evidence_log.append({

            "frame":
                int(
                    self.frame_idx
                ),

            "timestamp":
                int(
                    self.current_timestamp
                ),

            "ttc":
                float(
                    self.minimum_ttc
                ),

            "trust":
                float(
                    self.current_trust
                ),

            "runtime":
                self.runtime_state,

            "decision":
                self.decision_action,

            "collision":
                bool(
                    self.alert_collision
                ),

            "near_miss":
                bool(
                    self.near_miss
                ),

            "failed":
                bool(
                    self.failed_test
                )
        })

    def evaluate_scene_risk(
        self,
        frame
    ):

        self.minimum_ttc = 999.0

        max_unc = 0.0

        objects = frame["objects"]

        if self.active_track_id is not None:

            objects = [

                o for o in objects

                if o["track_id"]
                ==
                self.active_track_id
            ]

        for obj in objects:

            ttc = self.compute_ttc(obj)

            if np.isfinite(ttc):

                self.minimum_ttc = min(
                    self.minimum_ttc,
                    ttc
                )

            unc = self.estimate_track_uncertainty(
                obj["track_id"]
            )

            max_unc = max(
                max_unc,
                unc
            )

        ood_score = 0.0
        ood_flag = False

        if self.alert_ood:

            ood_score = float(
                self.alert_ood[0]
            )

            ood_flag = bool(
                self.alert_ood[1]
            )

        self.current_trust = (
            1.0
            - 0.2 * ood_score
            - 0.3 * max_unc
        )

        if ood_flag:
            self.current_trust -= 0.30

        self.current_trust = float(

            np.clip(
                self.current_trust,
                0,
                1
            )
        )

        if self.current_trust < 0.25:

            self.runtime_state = "UNSAFE"

        elif self.current_trust < 0.50:

            self.runtime_state = "WARNING"

        elif self.current_trust < 0.80:

            self.runtime_state = "DEGRADED"

        else:

            self.runtime_state = "NORMAL"

        if self.minimum_ttc < 2:

            self.decision_action = "STOP"

        elif self.minimum_ttc < 4:

            self.decision_action = "YIELD"

        elif self.minimum_ttc < 8:

            self.decision_action = "SLOW"

        else:

            self.decision_action = "GO"

        self.near_miss = (
            self.minimum_ttc < 1.0
            and
            not self.alert_collision
        )

        self.failed_test = False

        if self.alert_collision:

            self.failed_test = True

        if (
            self.minimum_ttc < 2
            and
            self.decision_action == "GO"
        ):
            self.failed_test = True

    def compute_ttc(self, obj):

        pts = np.asarray(obj["polygon"], dtype=float)

        if len(pts) < 3:
            return np.inf

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if path is None:
            return np.inf

        if self.current_waypoint_idx >= len(path.waypoints):
            return np.inf

        # ------------------------------------
        # Dataset frame period
        # ------------------------------------
        if len(tracks) < 2:
            return np.inf

        frame_idx = min(
            self.frame_idx,
            len(tracks) - 1
        )

        if frame_idx == 0:

            dt = (
                tracks[1]["timestamp"]
                - tracks[0]["timestamp"]
            ) / 1000.0

        else:

            dt = (
                tracks[frame_idx]["timestamp"]
                - tracks[frame_idx - 1]["timestamp"]
            ) / 1000.0

        if dt <= 0:
            return np.inf

        # ------------------------------------
        # Robot position
        # ------------------------------------
        robot_pos = np.array(
            [self.agent_x, self.agent_y],
            dtype=float
        )

        obj_pos = np.mean(
            pts,
            axis=0
        )

        # ------------------------------------
        # Path heading
        # ------------------------------------
        tx, ty = path.waypoints[
            self.current_waypoint_idx
        ]

        heading = np.array(
            [
                tx - self.agent_x,
                ty - self.agent_y
            ],
            dtype=float
        )

        heading_norm = np.linalg.norm(
            heading
        )

        if heading_norm < 1e-6:
            return np.inf

        heading /= heading_norm

        # ------------------------------------
        # Only objects in front
        # ------------------------------------
        rel_pos = obj_pos - robot_pos

        forward_distance = np.dot(
            rel_pos,
            heading
        )

        if forward_distance <= 0:
            return np.inf

        # ------------------------------------
        # Robot velocity
        # path.speed = pixels per dataset frame
        # convert to pixels/sec
        # ------------------------------------
        robot_speed = (
            path.speed *
            self.agent_speed_factor
        ) / dt

        robot_vel = (
            heading *
            robot_speed
        )

        # ------------------------------------
        # Object velocity
        # ------------------------------------
        obj_vel = np.asarray(
            obj.get(
                "velocity_world",
                [0.0, 0.0]
            ),
            dtype=float
        )

        # ------------------------------------
        # Relative closing speed
        # ------------------------------------
        rel_vel = obj_vel - robot_vel

        closing_speed = -np.dot(
            rel_vel,
            heading
        )

        if closing_speed <= 0:
            return np.inf

        # TTC (seconds)
        return (
            forward_distance
            / closing_speed
        )


    def snap_agent_to_active_path(self):

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if path is None:
            return

        if len(path.waypoints) < 2:
            return

        # move robot to path start

        self.agent_x = float(
            path.waypoints[0][0]
        )

        self.agent_y = float(
            path.waypoints[0][1]
        )

        self.agent_start_x = self.agent_x
        self.agent_start_y = self.agent_y

        self.current_waypoint_idx = 1

    def estimate_track_uncertainty(
        self,
        track_id
    ):

        frame = self.frame_idx

        hist = TRACK_UNCERTAINTY.get(
            track_id,
            []
        )

        if len(hist) < 4:
            return 0.0

        indexed = TRACK_HISTORY.get(
            track_id,
            []
        )

        current_idx = None

        for i, h in enumerate(indexed):
            if h[0] == frame:
                current_idx = i
                break

        if current_idx is None:
            return 1.0

        start = max(
            0,
            current_idx - 4
        )

        end = current_idx + 1

        recent = hist[start:end]

        if len(recent) < 3:
            return 0.0

        centers = np.asarray([
            h["center"]
            for h in recent
        ])

        areas = np.asarray([
            h["area"]
            for h in recent
        ])

        pos_var = np.linalg.norm(
            np.std(
                centers,
                axis=0
            )
        )

        area_cv = (
            np.std(areas)
            /
            (
                np.mean(areas) + 1e-6
            )
        )

        unc = (
            0.7 * min(1.0, pos_var / 15.0)
            +
            0.3 * min(1.0, area_cv)
        )

        return float(
            np.clip(
                unc,
                0.0,
                1.0
            )
        )

    def draw_robot_path(self):

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if path is None:
            return

        if len(path.waypoints) < 1:
            return

        xs = [p[0] for p in path.waypoints]
        ys = [p[1] for p in path.waypoints]

        line, = self.ax.plot(
            xs,
            ys,
            "--",
            color="yellow",
            linewidth=2,
            alpha=0.8,
            zorder=40
        )

        self.box_artists.append(line)

        pts = self.ax.scatter(
            xs,
            ys,
            s=60,
            c="red",
            zorder=45
        )

        self.box_artists.append(pts)

        #
        # current target waypoint
        #

        if (
            self.current_waypoint_idx < len(path.waypoints)
        ):

            tx, ty = path.waypoints[
                self.current_waypoint_idx
            ]

            target = self.ax.scatter(
                [tx],
                [ty],
                s=150,
                c="lime",
                marker="*",
                zorder=50
            )

            self.box_artists.append(target)

    def draw_entry_points(self):

        for artist in self.entry_artists:

            try:
                artist.remove()
            except:
                pass

        self.entry_artists.clear()

        for entry_id, (x, y) in ENTRY_POINTS.items():

            ellipse = Ellipse(
                (x, y),
                width=120,
                height=80,
                fill=False,
                edgecolor="orange",
                linewidth=2.5,
                alpha=0.8
            )

            self.ax.add_patch(
                ellipse
            )

            self.entry_artists.append(
                ellipse
            )

            label = self.ax.text(
                x,
                y,
                str(entry_id),
                color="orange",
                fontsize=12,
                weight="bold",
                ha="center",
                va="center"
            )

            self.entry_artists.append(
                label
            )
    # ==================================================
    # Mouse interaction
    # ==================================================

    def on_press(self, event):

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if self.path_edit_mode:

            path = DEFAULT_PATHS.get(
                self.current_path_name
            )

            if path is None:
                return

            # Left click -> add waypoint
            if event.button == 1:

                path.waypoints.append(
                    (
                        float(event.xdata),
                        float(event.ydata)
                    )
                )

            # Right click -> delete nearest waypoint
            elif event.button == 3:

                if len(path.waypoints):

                    px = float(event.xdata)
                    py = float(event.ydata)

                    nearest_idx = min(
                        range(len(path.waypoints)),
                        key=lambda i:
                        (
                            (path.waypoints[i][0]-px)**2 +
                            (path.waypoints[i][1]-py)**2
                        )
                    )

                    path.waypoints.pop(nearest_idx)

            self.update_frame(self.frame_idx)

            return

        dx = event.xdata - self.agent_x
        dy = event.ydata - self.agent_y

        if dx * dx + dy * dy < 20 * 20:
            self.dragging = True

    def on_motion(self, event):

        if not self.dragging:
            return

        if event.xdata is None or event.ydata is None:
            return

        self.agent_x = float(event.xdata)
        self.agent_y = float(event.ydata)

        self.vehicle_dot.set_data(
            [self.agent_x],
            [self.agent_y]
        )

        self.draw_idle()

    def on_release(self, event):

        if not self.dragging:
            return

        self.dragging = False

        self.agent_start_x = self.agent_x
        self.agent_start_y = self.agent_y

    def draw_active_trajectory(self):

        if self.trajectory_artist:

            try:
                self.trajectory_artist.remove()
            except:
                pass

            self.trajectory_artist = None

        self.current_traj_len = 0

        tid = self.active_track_id

        if tid is None:
            return

        history = TRACK_HISTORY.get(
            tid,
            []
        )

        history = [
            h
            for h in history
            if h[0] <= self.frame_idx
        ]

        self.current_traj_len = len(
            history
        )

        if len(history) < 2:
            return

        x = [p[1] for p in history]
        y = [p[2] for p in history]

        self.trajectory_artist, = self.ax.plot(
            x,
            y,
            color="yellow",
            linewidth=3,
            alpha=0.9,
            zorder=50
        )

    # ==================================================
    # Agent movement
    # ==================================================

    def update_agent(self):

        if self.dragging:
            return

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if path is None:
            return

        wp = path.waypoints

        if len(wp) < 2:
            return

        # path changed or robot invalid
        if (
            self.current_waypoint_idx <= 0
            or
            self.current_waypoint_idx >= len(wp)
        ):
            self.snap_agent_to_active_path()

        #
        # robot somehow not on path
        #

        if self.current_waypoint_idx <= 0:
            self.current_waypoint_idx = 1

        if self.current_waypoint_idx >= len(wp):
            return

        speed = (
            path.speed *
            self.agent_speed_factor
        )

        #
        # consume full motion budget
        #

        remaining = speed

        while remaining > 0:

            if self.current_waypoint_idx >= len(wp):
                break

            tx, ty = wp[
                self.current_waypoint_idx
            ]

            dx = tx - self.agent_x
            dy = ty - self.agent_y

            dist = math.hypot(
                dx,
                dy
            )

            if dist < 1.0:

                self.current_waypoint_idx += 1
                continue

            if dist <= remaining:

                self.agent_x = tx
                self.agent_y = ty

                remaining -= dist

                self.current_waypoint_idx += 1

                continue

            self.agent_x += (
                dx / dist
            ) * remaining

            self.agent_y += (
                dy / dist
            ) * remaining

            break
    # ==================================================
    # Main frame update
    # ==================================================

    def update_frame(self, idx):

        self.frame_idx = idx

        frame = tracks[idx]

        self.current_timestamp = (
            frame["timestamp"]
        )

        self.load_bev_image(
            self.current_timestamp
        )

        self.clear_artists()

        self.draw_objects(frame)

        self.update_agent()

        self.vehicle_dot.set_data(
            [self.agent_x],
            [self.agent_y]
        )

        self.compute_scene_info(frame)

        self.lookup_predictions()

        self.evaluate_scene_risk(
            frame
        )

        self.log_evidence()

        self.draw_entry_points()

        self.draw_active_trajectory()

        self.draw_robot_path()

        self.draw_idle()

    def get_relevant_cameras(self):

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if path is None:
            return []

        wp = path.waypoints

        if len(wp) < 2:
            return []

        cams = set()

        #
        # robot position
        #

        for c in getcamid(
            (
                self.agent_y,
                self.agent_x
            )
        ):
            cams.add(str(c))

        #
        # sample future route
        #

        px = self.agent_x
        py = self.agent_y

        for idx in range(
            self.current_waypoint_idx,
            len(wp)
        ):

            tx, ty = wp[idx]

            seg_len = math.hypot(
                tx - px,
                ty - py
            )

            n = max(
                2,
                int(seg_len / 20)
            )

            for k in range(n + 1):

                a = k / n

                sx = px + a * (
                    tx - px
                )

                sy = py + a * (
                    ty - py
                )

                found = getcamid(
                    (sy, sx)
                )

                for cam in found:

                    cams.add(
                        str(cam)
                    )

            px = tx
            py = ty

        return sorted(cams)

    def compute_scene_info(self, frame):

        self.alert_collision = False
        self.alert_ood = None

        vp = Point(
            self.agent_x,
            self.agent_y
        )

        self.relevant_cameras = (
            self.get_relevant_cameras()
        )

        #
        # display only
        #

        self.alert_cam = None

        if self.relevant_cameras:

            self.alert_cam = ",".join(
                self.relevant_cameras
            )

        #
        # worst OOD across relevant cameras
        #

        ts = int(
            self.current_timestamp
        )

        worst_score = -1

        for cam in self.relevant_cameras:

            scores, keys = load_ood(cam)

            if not keys:
                continue

            nearest_key = min(
                keys,
                key=lambda k:
                abs(int(k) - ts)
            )

            result = scores.get(
                nearest_key
            )

            if result is None:
                continue

            score = float(
                result[0]
            )

            if score > worst_score:

                worst_score = score

                self.alert_ood = result

        #
        # collision
        #

        for obj in frame["objects"]:

            pts = obj["polygon"]

            if len(pts) < 3:
                continue

            try:

                if Polygon(
                    pts
                ).buffer(
                    VEHICLE_RADIUS
                ).contains(vp):

                    self.alert_collision = True
                    break

            except Exception:
                pass

    def lookup_predictions(self):

        self.route_info = None
        self.mode_info = None

        if (
            self.active_track_id is None
            or self.current_timestamp is None
        ):
            return

        pred = (
            prediction_lookup
            .get(
                int(self.current_timestamp),
                {}
            )
            .get(
                int(self.active_track_id)
            )
        )

        if pred is None:
            return

        self.route_info = pred.get(
            "route"
        )

        self.mode_info = pred

    def draw_objects(self, frame):

        self.track_ids_current = []

        active = self.active_track_id

        for obj in frame["objects"]:

            tid = obj["track_id"]

            self.track_ids_current.append(
                tid
            )

            pts = obj["polygon"]

            if len(pts) < 3:
                continue

            pts = np.asarray(pts)

            is_active = (
                tid == active
            )

            line, = self.ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=(
                    "lime"
                    if is_active
                    else "cyan"
                ),
                lw=(
                    3
                    if is_active
                    else 1.5
                ),
                alpha=(
                    1.0
                    if is_active
                    else 0.75
                )
            )

            self.box_artists.append(line)

            self.box_artists.append(
                self.ax.text(
                    pts[:, 0].mean(),
                    pts[:, 1].mean(),
                    str(tid),
                    color="yellow",
                    fontsize=8
                )
            )

    def clear_artists(self):

        for a in self.box_artists:
            a.remove()

        self.box_artists.clear()

    def load_bev_image(self, timestamp):

        mask_file = mask_lookup.get(timestamp)

        if mask_file is None:
            return

        img = cv2.imread(
            os.path.join(
                MASK_FOLDER,
                mask_file
            ),
            cv2.IMREAD_GRAYSCALE
        )

        self.img.set_data(img)

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
        self.new_path_btn = QPushButton("New Path")
        self.save_path_btn = QPushButton("Save Paths")
        self.load_path_btn = QPushButton("Load Paths")        
        self.path_select = QComboBox()
        for name in DEFAULT_PATHS:
            self.path_select.addItem(name) 
        self.slider.setRange(0, NUM_FRAMES-1)
        self.label_frame = QLabel("Frame: 0")

        # Info panel
        self.label_result_banner = QLabel("PASS")
        self.label_result_banner.setAlignment(            Qt.AlignmentFlag.AlignCenter        )
        self.label_result_banner.setStyleSheet(
            """
            background:#2e7d32;
            color:white;
            font-size:18px;
            font-weight:bold;
            padding:8px;
            border-radius:4px;
            """
        )
        self.label_action = QLabel("Decision: GO")
        self.label_runtime = QLabel("Runtime: NORMAL")
        self.label_ttc = QLabel("Min TTC: -")
        self.label_trust = QLabel("Trust: -")
        self.label_status = QLabel("Collision: NO")
        self.label_object = QLabel("Forklift: -")
        self.label_route = QLabel("Route: -")
        self.label_route_prob = QLabel("Route Prob: -")
        self.label_route_unc = QLabel("Route Uncertainty: -")
        self.label_mode = QLabel("Maneuver: -")
        self.label_mode_prob = QLabel("Maneuver Confidence: -")
        self.label_cam = QLabel("Camera: -")
        self.label_ood = QLabel("OOD: -")
        self.label_frame = QLabel("Frame: 0")

        self.object_select = QComboBox()
        self.edit_path_btn = QPushButton("Edit Path")
        self.speed_select = QComboBox()
        self.speed_select.addItems([
            "0.5x",
            "1x",
            "2x",
            "5x",
            "10x"
        ])
        self.speed_select.setCurrentText("1x")

       

        ctrl = QVBoxLayout()

        # ==================================================
        # Playback
        # ==================================================

        ctrl.addWidget(self.load_btn)
        ctrl.addWidget(self.play_btn)
        ctrl.addWidget(self.reset_btn)
        ctrl.addWidget(self.label_frame)
        ctrl.addWidget(self.slider)

        # ==================================================
        # Path
        # ==================================================

        ctrl.addWidget(QLabel("AMR Path"))
        ctrl.addWidget(self.path_select)
        ctrl.addWidget(self.new_path_btn)
        ctrl.addWidget(self.save_path_btn)
        ctrl.addWidget(self.load_path_btn)
        ctrl.addWidget(self.edit_path_btn)

        ctrl.addWidget(QLabel("AMR Speed"))
        ctrl.addWidget(self.speed_select)

        # ==================================================
        # Forklift Selection
        # ==================================================

        ctrl.addWidget(QLabel("Active Forklift"))
        ctrl.addWidget(self.object_select)

        # ==================================================
        # Safety Summary
        # ==================================================

        ctrl.addSpacing(10)
        ctrl.addWidget(self.label_result_banner)

        ctrl.addWidget(self.label_action)
        ctrl.addWidget(self.label_runtime)
        ctrl.addWidget(self.label_ttc)
        ctrl.addWidget(self.label_trust)
        ctrl.addWidget(self.label_status)

        # ==================================================
        # Prediction
        # ==================================================

        ctrl.addSpacing(10)

        route_header = QLabel("Prediction")
        route_header.setStyleSheet(
            "font-weight:bold;"
        )
        ctrl.addWidget(route_header)

        ctrl.addWidget(self.label_object)
        ctrl.addWidget(self.label_route)
        ctrl.addWidget(self.label_route_prob)
        ctrl.addWidget(self.label_route_unc)
        ctrl.addWidget(self.label_mode)
        ctrl.addWidget(self.label_mode_prob)

        # ==================================================
        # Diagnostics
        # ==================================================

        ctrl.addSpacing(10)

        diag_header = QLabel("Diagnostics")
        diag_header.setStyleSheet(
            "font-weight:bold;"
        )
        ctrl.addWidget(diag_header)

        ctrl.addWidget(self.label_cam)
        ctrl.addWidget(self.label_ood)

        ctrl.addStretch()

        main = QHBoxLayout()
        main.addWidget(self.viewer, 1)
        c = QWidget(); c.setLayout(ctrl); c.setFixedWidth(250)
        main.addWidget(c)

        root = QWidget(); root.setLayout(main)
        self.setCentralWidget(root)

        self.play_btn.clicked.connect(self.toggle)
        self.reset_btn.clicked.connect(self.reset)
        self.slider.valueChanged.connect(self.seek)
        self.speed_select.currentTextChanged.connect(self.change_speed)    
        self.object_select.currentTextChanged.connect(self.change_object)        
        self.path_select.currentTextChanged.connect(self.change_path)
        self.edit_path_btn.clicked.connect(self.toggle_path_edit)
        self.new_path_btn.clicked.connect(self.create_new_path)
        self.save_path_btn.clicked.connect(self.save_path_library)
        self.load_path_btn.clicked.connect(self.load_path_library)

    def save_path_library(self):
        save_paths()


    def load_path_library(self):
        load_paths()
        self.path_select.blockSignals(True)
        self.path_select.clear()
        for name in DEFAULT_PATHS:
            self.path_select.addItem(name)
        self.path_select.blockSignals(False)
        if self.path_select.count():

            self.path_select.setCurrentIndex(0)

            self.viewer.current_path_name = (
                self.path_select.currentText()
            )

            self.viewer.snap_agent_to_active_path()

            self.viewer.update_frame(
                self.viewer.frame_idx
            )


    def create_new_path(self):
        idx = 1
        while f"Path_{idx}" in DEFAULT_PATHS:
            idx += 1
        name = f"Path_{idx}"
        DEFAULT_PATHS[name] = RobotPath(name)
        self.path_select.addItem(name)
        self.path_select.setCurrentText(name)

    def toggle_path_edit(self):

        self.viewer.path_edit_mode = (
            not self.viewer.path_edit_mode
        )

        path = DEFAULT_PATHS[
            self.viewer.current_path_name
        ]

        if self.viewer.path_edit_mode:

            reply = len(path.waypoints)

            if reply == 0:

                path.waypoints.clear()

            self.edit_path_btn.setText(
                "Finish Path"
            )

        else:

            self.edit_path_btn.setText(
                "Edit Path"
            )

            if len(path.waypoints):

                self.viewer.agent_x = (
                    path.waypoints[0][0]
                )

                self.viewer.agent_y = (
                    path.waypoints[0][1]
                )

                self.viewer.current_waypoint_idx = 1

            save_paths()

            self.viewer.update_frame(
                self.viewer.frame_idx
            )

    def change_path(self, name):

        self.viewer.current_path_name = name

        path = DEFAULT_PATHS.get(name)

        if path is None:
            return

        if len(path.waypoints):
            self.viewer.snap_agent_to_active_path()

        self.viewer.current_waypoint_idx = 1

        self.viewer.update_frame(
            self.viewer.frame_idx
        )

        self.statusBar().showMessage(
            f"{name}: "
            f"{len(path.waypoints)} "
            f"waypoints"
        )

    def change_object(self, txt):

        try:
            self.viewer.active_track_id = int(txt)

        except ValueError:
            return

        self.viewer.lookup_predictions()

        self.refresh_info_panel()

        self.viewer.draw_idle()

    def next_frame(self):

        current_track = (
            self.viewer.active_track_id
        )
        i = (
            self.viewer.frame_idx + 1
        ) % NUM_FRAMES
        self.viewer.update_frame(i)

        self.viewer.active_track_id = (
            current_track
        )
        self.viewer.lookup_predictions()
        self.refresh_object_combo()

        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)

        ts = self.viewer.current_timestamp

        self.label_frame.setText(
            f"Frame: {i} | TS: {ts}"
        )

        self.refresh_info_panel()

    def refresh_info_panel(self):

        v = self.viewer

        route = v.route_info or {}
        mode = v.mode_info or {}

        #
        # Result Banner
        #

        if v.alert_collision:

            self.label_result_banner.setText(
                "FAIL - COLLISION"
            )

            color = "#c62828"

        elif v.failed_test:

            self.label_result_banner.setText(
                "FAIL"
            )

            color = "#ef6c00"

        elif v.near_miss:

            self.label_result_banner.setText(
                "WARNING"
            )

            color = "#f9a825"

        else:

            self.label_result_banner.setText(
                "PASS"
            )

            color = "#2e7d32"

        self.label_result_banner.setStyleSheet(
            f"""
            background:{color};
            color:white;
            font-size:18px;
            font-weight:bold;
            padding:8px;
            border-radius:4px;
            """
        )

        #
        # Safety summary
        #

        self.label_action.setText(
            f"Decision : {v.decision_action}"
        )

        self.label_runtime.setText(
            f"Runtime : {v.runtime_state}"
        )

        if np.isfinite(v.minimum_ttc):
            self.label_ttc.setText(
                f"TTC : {v.minimum_ttc:.1f}s"
            )
        else:
            self.label_ttc.setText(
                "TTC : N/A"
            )

        self.label_trust.setText(
            f"Trust : {v.current_trust:.2f}"
        )

        if v.alert_collision:

            self.label_status.setText(
                "Collision : YES"
            )

        elif v.near_miss:

            self.label_status.setText(
                "Collision : NEAR MISS"
            )

        else:

            self.label_status.setText(
                "Collision : NO"
            )

        #
        # Forklift information
        #

        self.label_object.setText(
            f"Forklift ID : {v.active_track_id}"
        )

        if route:

            self.label_route.setText(
                f"Route : {route.get('from','?')} → {route.get('to','?')}"
            )

            prob = route.get("prob")

            if prob is not None:

                self.label_route_prob.setText(
                    f"Route Prob : {prob:.2f}"
                )

            else:

                self.label_route_prob.setText(
                    "Route Prob : N/A"
                )

            unc = route.get("uncertainty")

            if unc is not None:

                self.label_route_unc.setText(
                    f"Route Unc : {unc:.2f}"
                )

            else:

                self.label_route_unc.setText(
                    "Route Unc : N/A"
                )

        else:

            self.label_route.setText(
                "Route : N/A"
            )

            self.label_route_prob.setText(
                "Route Prob : N/A"
            )

            self.label_route_unc.setText(
                "Route Unc : N/A"
            )

        self.label_mode.setText(
            f"Maneuver : {mode.get('mode','N/A')}"
        )

        conf = mode.get(
            "display_confidence"
        )

        if conf is not None:

            self.label_mode_prob.setText(
                f"Confidence : {conf:.2f}"
            )

        else:

            self.label_mode_prob.setText(
                "Confidence : N/A"
            )

        #
        # Diagnostics
        #

        cams = getattr(
            v,
            "relevant_cameras",
            []
        )

        if cams:

            self.label_cam.setText(
                "Camera(s) : "
                + ",".join(cams)
            )

        else:

            self.label_cam.setText(
                "Camera(s) : NONE"
            )

        if v.alert_ood:
            score = float(v.alert_ood[0])
            flag = bool(v.alert_ood[1])

            status = (
                "ANOMALOUS"
                if flag
                else
                "NORMAL"
            )
            self.label_ood.setText(
                f"OOD : {score:.9f} | {status}"
            )
        else:

            self.label_ood.setText(
                "OOD : N/A"
            )

        #
        # Color coding
        #

        if v.minimum_ttc < 2:

            self.label_ttc.setStyleSheet(
                "color:#ff4444;font-weight:bold;"
            )

        elif v.minimum_ttc < 4:

            self.label_ttc.setStyleSheet(
                "color:#ffb300;font-weight:bold;"
            )

        else:

            self.label_ttc.setStyleSheet(
                "color:#66bb6a;font-weight:bold;"
            )

        if v.current_trust < 0.4:

            self.label_trust.setStyleSheet(
                "color:#ff4444;font-weight:bold;"
            )

        elif v.current_trust < 0.7:

            self.label_trust.setStyleSheet(
                "color:#ffb300;font-weight:bold;"
            )

        else:

            self.label_trust.setStyleSheet(
                "color:#66bb6a;font-weight:bold;"
            )

    def refresh_object_combo(self):

        visible = sorted(
            set(self.viewer.track_ids_current)
        )

        selected = self.viewer.active_track_id

        self.object_select.blockSignals(True)

        self.object_select.clear()

        for tid in visible:
            self.object_select.addItem(str(tid))

        # keep previously selected track
        if selected in visible:

            idx = self.object_select.findText(
                str(selected)
            )

            if idx >= 0:
                self.object_select.setCurrentIndex(idx)

        elif visible:

            selected = visible[0]

            self.viewer.active_track_id = selected

            self.object_select.setCurrentIndex(0)

        self.object_select.blockSignals(False)


    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.timer.start()
            self.play_btn.setText("⏸ Pause")

    def reset(self):
        self.timer.stop()
        path = DEFAULT_PATHS[
            self.viewer.current_path_name
        ]
        if len(path.waypoints):
            self.viewer.agent_x = path.waypoints[0][0]
            self.viewer.agent_y = path.waypoints[0][1]
            self.viewer.current_waypoint_idx = 1
        self.viewer.frame_idx = 0
        self.viewer.update_frame(0)

    def seek(self, value):

        self.viewer.update_frame(value)

        self.refresh_object_combo()

        ts = self.viewer.current_timestamp

        self.label_frame.setText(
            f"Frame: {value} | TS: {ts}"
        )

        self.refresh_info_panel()


    def change_speed(self, text):

        factor = float(
            text.replace("x", "")
        )

        self.viewer.agent_speed_factor = factor


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    sys.exit(app.exec())
