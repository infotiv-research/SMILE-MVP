# ==========================================================
# TUVE MVP FULL PyQt6 Port (Feature Complete + Info Panel)
# ==========================================================

import sys, os, json, gzip, cv2, pickle 
import numpy as np
import math
from pathlib import Path
from typing import Tuple
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
import joblib

from scenario_supervisor import evaluate_scenario_1

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel,
    QFileDialog, QComboBox, QCheckBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Polygon as MplPolygon


def _consume_cli_option(flag):
    """Remove and return a simple option so Qt does not receive it."""
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    return value


SCENARIO_REPLAY_FILE = _consume_cli_option("--scenario-replay")
SCENARIO_REPLAY_ID = _consume_cli_option("--scenario-id")
REFERENCE_PATHS_FILE = _consume_cli_option("--reference-paths")
SCENARIO_REPLAY = None
SCENARIO_FRAME_META = {}
# In "All paths" replay mode this holds the ten independent AMR route variants
# associated with the active recorded forklift window.
SCENARIO_ALL_PATHS = []

# ==========================================================
# CONFIG (unchanged)
# ==========================================================

# Paths and runtime values come from mvp-config.json
# (run this script from the repository root)
CONFIG_FILE = "mvp-config.json"

with open(CONFIG_FILE, "r") as _f:
    CFG = json.load(_f)

REPO_ROOT = Path(__file__).resolve().parent

def repo_path(path_value):
    """Resolve configuration paths from the repository, not the dataset symlink."""
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json_file(path):
    """Read plain or gzip-compressed JSON replay artifacts."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as source:
        return json.load(source)

DATA_ROOT = repo_path(CFG["paths"]["dataset_root"])
EXTRA_PATHS = CFG["extra_paths"]

MASK_FOLDER = str(DATA_ROOT / EXTRA_PATHS["mask_folder"])
SAVE_PATH = str(repo_path(CFG["paths"]["save_path"]))
PRED_PATH = str(repo_path(EXTRA_PATHS["predictions"]))
OOD_ROOT = str(repo_path(CFG["paths"]["ood_root"]))
CAMERA_LOOKUP_PATH = str(repo_path(CFG["paths"]["camera_lookup_table"]))
OCCUPANCY_PATH = str(repo_path(EXTRA_PATHS["occupancy_projection"]))
PATH_DB = str(repo_path(EXTRA_PATHS["robot_paths"]))
REFERENCE_PATHS_PATH = repo_path(
    REFERENCE_PATHS_FILE or "JSON-data/robot_reference_paths.json"
)
INTERSECTION_AREA_PATH = repo_path("JSON-data/intersection_area.json")
ROBOT_OPERATING_AREA_PATH = repo_path("JSON-data/robot_operating_area.json")

VEHICLE_RADIUS = CFG["runtime"]["vehicle_radius"]
MAX_DISTANCE = CFG["runtime"]["max_distance"]

OOD_CACHE = {}

INTERSECTION_POINTS = []
if INTERSECTION_AREA_PATH.exists():
    with open(INTERSECTION_AREA_PATH, "r") as _intersection_file:
        INTERSECTION_POINTS = json.load(_intersection_file).get("polygon_image_px", [])

ROBOT_OPERATING_AREA_POINTS = []
if ROBOT_OPERATING_AREA_PATH.exists():
    with open(ROBOT_OPERATING_AREA_PATH, "r") as _operating_area_file:
        ROBOT_OPERATING_AREA_POINTS = json.load(_operating_area_file).get(
            "polygon_image_px", []
        )

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


def build_station_points(reference_tracks, reference_predictions):
    """Locate every trajectory-prediction station from the full recording."""
    first_centres = {}
    samples = {}
    for frame in reference_tracks:
        timestamp = int(frame["timestamp"])
        predictions = reference_predictions.get(timestamp, {})
        for obj in frame["objects"]:
            polygon = np.asarray(obj.get("polygon", []), dtype=float)
            if len(polygon) < 3:
                continue
            track_id = int(obj["track_id"])
            first_centres.setdefault(track_id, tuple(np.mean(polygon, axis=0)))
            route = predictions.get(track_id, {}).get("route")
            if not route or route.get("from") is None:
                continue
            samples.setdefault(route["from"], []).append(first_centres[track_id])
    return {
        entry_id: tuple(np.mean(np.asarray(points), axis=0))
        for entry_id, points in samples.items()
    }


# Build stations before an optional scenario replay replaces the full track set.
STATIC_ENTRY_POINTS = build_station_points(tracks, prediction_lookup)


def load_scenario_replay(path, instance_id=None):
    """Load generated Scenario 1 data as a drop-in tracks/predictions source."""
    payload = load_json_file(path)

    instances = payload.get("instances", [])
    if not instances:
        raise ValueError(f"No replay instances in {path}")

    instance = next(
        (item for item in instances if item.get("id") == instance_id),
        instances[0] if instance_id is None else None
    )
    if instance is None:
        available = ", ".join(item.get("id", "?") for item in instances)
        raise ValueError(f"Scenario instance {instance_id!r} not found; available: {available}")

    replay_tracks = []
    replay_predictions = {}
    metadata = {}
    for frame in instance["frames"]:
        timestamp = int(frame["timestamp"])
        replay_tracks.append({"timestamp": timestamp, "objects": frame.get("objects", [])})
        replay_predictions[timestamp] = {
            int(pred["track_id"]): pred for pred in frame.get("predictions", [])
        }
        metadata[timestamp] = frame

    instance = dict(instance)
    instance["intersection_polygon_px"] = payload["intersection_polygon_px"]
    instance["replay_scenario"] = payload.get("scenario", "")
    return instance, replay_tracks, replay_predictions, metadata


if SCENARIO_REPLAY_FILE:
    scenario_path = repo_path(SCENARIO_REPLAY_FILE)
    (
        SCENARIO_REPLAY,
        tracks,
        prediction_lookup,
        SCENARIO_FRAME_META,
    ) = load_scenario_replay(scenario_path, SCENARIO_REPLAY_ID)
    print(f"Replaying {SCENARIO_REPLAY['id']}: {SCENARIO_REPLAY['title']}")

# The normal dataset length was computed before an optional replay replaces
# ``tracks``. Always use the active source's length for sliders and playback.
NUM_FRAMES = len(tracks)

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


def rebuild_runtime_track_caches():
    """Rebuild per-track state after changing the active replay instance."""
    global TRACK_HISTORY, TRACK_UNCERTAINTY
    TRACK_HISTORY = {}
    TRACK_UNCERTAINTY = {}
    for frame_idx, frame in enumerate(tracks):
        for obj in frame["objects"]:
            track_id = obj["track_id"]
            polygon = np.asarray(obj["polygon"], dtype=np.float32)
            if len(polygon) < 3:
                continue
            center = np.mean(polygon, axis=0)
            TRACK_HISTORY.setdefault(track_id, []).append(
                (frame_idx, float(center[0]), float(center[1]))
            )
            TRACK_UNCERTAINTY.setdefault(track_id, []).append({
                "center": center,
                "area": cv2.contourArea(polygon),
                "frame": frame["timestamp"],
            })

# ==========================================================
# AUTO BUILD ENTRY POINTS FROM ROUTE PREDICTIONS
# ==========================================================

# Station circles are intentionally sourced from the full recording, so every
# trajectory-prediction class remains visible during short scenario replays.
ENTRY_POINTS = STATIC_ENTRY_POINTS

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


def load_reference_paths():
    """Load user-authored routes without mixing them into the live path library."""
    if not REFERENCE_PATHS_PATH.exists():
        return {}
    with open(REFERENCE_PATHS_PATH, "r") as reference_file:
        return json.load(reference_file)


def save_reference_paths(reference_paths):
    with open(REFERENCE_PATHS_PATH, "w") as reference_file:
        json.dump(reference_paths, reference_file, indent=2)


def reference_path_sort_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


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

def install_replay_path():
    """Install the active scenario AMR route without saving it to the library."""
    if not SCENARIO_REPLAY:
        return
    replay_path = RobotPath("Path_1")
    replay_path.waypoints = [
        (float(x), float(y))
        for x, y in SCENARIO_REPLAY["robot_path"]
    ]
    replay_path.speed = float(SCENARIO_REPLAY["robot_speed_px_per_frame"])
    DEFAULT_PATHS["Path_1"] = replay_path


def activate_scenario_replay(instance_id, show_all_paths=False):
    """Switch the global data source to another generated scenario instance."""
    global SCENARIO_REPLAY, tracks, prediction_lookup, SCENARIO_FRAME_META, NUM_FRAMES, SCENARIO_ALL_PATHS
    if not SCENARIO_REPLAY_FILE:
        return
    replay_path = repo_path(SCENARIO_REPLAY_FILE)
    if show_all_paths:
        all_instances = load_json_file(replay_path).get("instances", [])
        selected = next((item for item in all_instances if item.get("id") == instance_id), None)
        window_id = selected.get("window_id") if selected else None
        SCENARIO_ALL_PATHS = [
            item for item in all_instances if item.get("window_id") == window_id
        ]
        # Use the longest sibling replay so every independently moving AMR
        # has recorded forklift imagery for its complete visible route.
        if SCENARIO_ALL_PATHS:
            instance_id = max(SCENARIO_ALL_PATHS, key=lambda item: len(item.get("frames", [])))["id"]
    else:
        SCENARIO_ALL_PATHS = []
    (
        SCENARIO_REPLAY,
        tracks,
        prediction_lookup,
        SCENARIO_FRAME_META,
    ) = load_scenario_replay(replay_path, instance_id)
    NUM_FRAMES = len(tracks)
    rebuild_runtime_track_caches()
    install_replay_path()


install_replay_path()

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

        if len(ROBOT_OPERATING_AREA_POINTS) >= 3:
            self.operating_area_artist = MplPolygon(
                ROBOT_OPERATING_AREA_POINTS,
                closed=True,
                facecolor="limegreen",
                edgecolor="lime",
                linewidth=2.0,
                alpha=0.14,
                zorder=25,
                label="Robot operating area",
            )
            self.ax.add_patch(self.operating_area_artist)
            self.operating_area_label = self.ax.text(
                0.02, 0.035, "ROBOT OPERATING AREA", color="lime", fontsize=8,
                ha="left", va="bottom", weight="bold", zorder=60,
                transform=self.ax.transAxes,
            )
        else:
            self.operating_area_artist = None
            self.operating_area_label = None

        if len(INTERSECTION_POINTS) >= 3:
            self.intersection_artist = MplPolygon(
                INTERSECTION_POINTS,
                closed=True,
                facecolor="deepskyblue",
                edgecolor="cyan",
                linewidth=2.5,
                alpha=0.22,
                zorder=35,
                label="Intersection area",
            )
            self.ax.add_patch(self.intersection_artist)
            self.intersection_label = self.ax.text(
                0.02, 0.075, "INTERSECTION", color="cyan", fontsize=9,
                ha="left", va="bottom", weight="bold", zorder=60,
                transform=self.ax.transAxes,
            )
        else:
            self.intersection_artist = None
            self.intersection_label = None

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
        # A generated replay declares the forklift that caused the interaction.
        # Keep it selected instead of defaulting to the lowest visible ID.
        self.active_track_id = (
            int(SCENARIO_REPLAY["focus_track_id"])
            if SCENARIO_REPLAY and SCENARIO_REPLAY.get("focus_track_id") is not None
            else None
        )

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
        self.show_reference_paths = False
        self.reference_path_edit_mode = False
        self.reference_path_edit_name = None
        self.reference_path_edit_waypoints = []

        self.dragging = False
        self.parallel_robot_states = []
        self.parallel_last_frame = None
        self.use_precomputed_behavior = True
        self.safe_parking_active = False
        self.safe_parking_arrived = False
        self.safe_parking_target = None
        self.safe_parking_clear_frames = 0

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

            "intersection_reason":
                getattr(self, "intersection_reason", None),

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

    def path_enters_intersection(self):
        """Whether the remaining AMR route intersects the defined area."""
        if len(INTERSECTION_POINTS) < 3:
            return False
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None or len(path.waypoints) < 2:
            return False
        route = [(self.agent_x, self.agent_y)] + path.waypoints[
            self.current_waypoint_idx:
        ]
        return len(route) >= 2 and LineString(route).intersects(
            Polygon(INTERSECTION_POINTS)
        )

    def robot_time_to_intersection(self):
        """Approximate seconds before the AMR reaches the intersection edge."""
        if not self.path_enters_intersection():
            return np.inf
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None:
            return np.inf
        if len(tracks) < 2:
            return np.inf
        frame = min(self.frame_idx, len(tracks) - 1)
        prior = max(0, frame - 1)
        dt_s = (tracks[frame]["timestamp"] - tracks[prior]["timestamp"]) / 1000.0
        if dt_s <= 0:
            dt_s = 0.1
        speed_px_s = (path.speed * self.agent_speed_factor) / dt_s
        if speed_px_s <= 1e-6:
            return np.inf
        return Point(self.agent_x, self.agent_y).distance(
            Polygon(INTERSECTION_POINTS)
        ) / speed_px_s

    def observed_pixel_velocity(self, obj):
        """Estimate BEV velocity from consecutive recorded polygon centres."""
        if "velocity_px_s" in obj:
            return obj["velocity_px_s"]
        history = [
            item for item in TRACK_HISTORY.get(obj["track_id"], [])
            if item[0] <= self.frame_idx
        ]
        if len(history) < 2:
            return [0.0, 0.0]
        previous, current = history[-2:]
        dt_s = (tracks[current[0]]["timestamp"] - tracks[previous[0]]["timestamp"]) / 1000.0
        if dt_s <= 0:
            return [0.0, 0.0]
        return [
            (current[1] - previous[1]) / dt_s,
            (current[2] - previous[2]) / dt_s,
        ]

    def evaluate_scene_risk(
        self,
        frame
    ):

        self.minimum_ttc = 999.0

        max_unc = 0.0

        # TTC is a scene-safety quantity: any forklift with a short closing
        # time can require a protective action.  ``active_track_id`` remains
        # a presentation/control selection only; it must not exclude other
        # forklifts from the minimum-TTC decision.
        objects = frame["objects"]

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

        # Scenario 1 intersection entry/clearance supervisor. This applies to
        # the ordinary recording viewer as well as generated scenario replay.
        if len(INTERSECTION_POINTS) >= 3:
            # Safety is evaluated against the complete recorded scene.  A
            # replay's focus track is only a visualization/reporting aid;
            # every forklift in or approaching the intersection can delay
            # entry, require clearance, or trigger the fail-safe response.
            intersection_objects = frame["objects"]
            intersection_forklifts = []
            intersection_uncertainty = 0.0
            for obj in intersection_objects:
                pred = prediction_lookup.get(
                    int(self.current_timestamp), {}
                ).get(int(obj["track_id"]), {})
                intersection_forklifts.append({
                    "track_id": obj["track_id"],
                    "polygon": obj["polygon"],
                    "velocity_px_s": self.observed_pixel_velocity(obj),
                    "maneuver": pred.get("mode", "moving"),
                })
                intersection_uncertainty = max(
                    intersection_uncertainty,
                    self.estimate_track_uncertainty(obj["track_id"]),
                )

            replay_frame = SCENARIO_FRAME_META.get(
                int(self.current_timestamp), {}
            )
            scenario_uncertainty = max(
                intersection_uncertainty,
                float(replay_frame.get("perception_uncertainty", 0.0)),
            )
            intersection_result = evaluate_scenario_1(
                [self.agent_x, self.agent_y],
                intersection_forklifts,
                INTERSECTION_POINTS,
                ood_flag=ood_flag,
                uncertainty=scenario_uncertainty,
                robot_path_intersects_intersection=self.path_enters_intersection(),
                robot_entry_time_s=self.robot_time_to_intersection(),
            )
            self.intersection_reason = intersection_result["reason"]
            # Keep an already more restrictive TTC decision, otherwise apply
            # the intersection-specific entry, clearance or fail-safe action.
            action_priority = {
                "GO": 0, "SLOW": 1, "YIELD": 2,
                "CLEAR_INTERSECTION": 2, "STOP": 3,
            }
            if action_priority.get(intersection_result["action"], 0) >= action_priority.get(
                self.decision_action, 0
            ):
                self.decision_action = intersection_result["action"]

        # Scenario 2 head-on mitigation. Move to the nearest boundary of the
        # approved operating area and remain parked until the opposing forklift
        # has been absent for a short confirmation interval.
        scenario_2_replay = SCENARIO_REPLAY and str(
            SCENARIO_REPLAY.get("replay_scenario", "")
        ).startswith("Scenario 2")
        if scenario_2_replay and self.head_on_forklift_present(frame):
            self.begin_safe_parking()
            if self.safe_parking_active:
                self.safe_parking_clear_frames = 0
                self.decision_action = "PARK_SAFE"
        elif self.safe_parking_active:
            self.safe_parking_clear_frames += 1
            if self.safe_parking_arrived and self.safe_parking_clear_frames >= 10:
                self.resume_from_safe_parking()
                self.decision_action = "GO"
            else:
                self.decision_action = "PARK_SAFE"

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

        if SCENARIO_ALL_PATHS:
            colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]
            for index, replay_path in enumerate(SCENARIO_ALL_PATHS):
                waypoints = replay_path.get("robot_path", [])
                if len(waypoints) < 2:
                    continue
                xs = [point[0] for point in waypoints]
                ys = [point[1] for point in waypoints]
                line, = self.ax.plot(
                    xs, ys, "--", color=colors[index % len(colors)], linewidth=1.6,
                    alpha=0.7, zorder=40,
                )
                self.box_artists.append(line)
                state = next(
                    (item for item in self.parallel_robot_states
                     if item["replay"].get("id") == replay_path.get("id")),
                    None,
                )
                if state is None:
                    px, py = waypoints[0]
                else:
                    px, py = state["x"], state["y"]
                robot = self.ax.scatter(
                    [px], [py], s=55, color=colors[index % len(colors)], edgecolors="white",
                    linewidths=0.5, zorder=55,
                )
                self.box_artists.append(robot)
            return

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

    def draw_reference_paths(self):
        """Overlay all saved reference routes; the editable route is highlighted."""
        if not self.show_reference_paths:
            return
        reference_paths = load_reference_paths()
        if self.reference_path_edit_mode and self.reference_path_edit_name:
            reference_paths = dict(reference_paths)
            reference_paths[self.reference_path_edit_name] = {
                "waypoints": self.reference_path_edit_waypoints,
            }
        for name, definition in reference_paths.items():
            waypoints = definition.get("waypoints", [])
            if len(waypoints) < 2:
                continue
            editing = name == self.reference_path_edit_name
            xs = [point[0] for point in waypoints]
            ys = [point[1] for point in waypoints]
            line, = self.ax.plot(
                xs, ys, "-", color="yellow" if editing else "white",
                linewidth=3.0 if editing else 1.3,
                alpha=0.95 if editing else 0.45, zorder=46 if editing else 39,
            )
            self.box_artists.append(line)
            label = self.ax.text(
                xs[0], ys[0], name, color="yellow" if editing else "white",
                fontsize=7, weight="bold" if editing else "normal", zorder=47,
            )
            self.box_artists.append(label)
            if editing:
                points = self.ax.scatter(xs, ys, s=55, c="red", zorder=48)
                self.box_artists.append(points)

    @staticmethod
    def point_on_replay_path(waypoints, distance):
        """Nominal position for an independently executing replay AMR."""
        remaining = max(0.0, distance)
        for start, end in zip(waypoints, waypoints[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            if remaining <= length:
                return start[0] + dx * remaining / length, start[1] + dy * remaining / length
            remaining -= length
        return waypoints[-1][0], waypoints[-1][1]

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

        if self.reference_path_edit_mode:
            if event.button == 1:
                self.reference_path_edit_waypoints.append(
                    [float(event.xdata), float(event.ydata)]
                )
            elif event.button == 3 and self.reference_path_edit_waypoints:
                px, py = float(event.xdata), float(event.ydata)
                nearest = min(
                    range(len(self.reference_path_edit_waypoints)),
                    key=lambda index: (
                        (self.reference_path_edit_waypoints[index][0] - px) ** 2
                        + (self.reference_path_edit_waypoints[index][1] - py) ** 2
                    ),
                )
                self.reference_path_edit_waypoints.pop(nearest)
            self.update_frame(self.frame_idx)
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

        if SCENARIO_ALL_PATHS:
            self.vehicle_dot.set_data([], [])
        else:
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

    def head_on_forklift_present(self, frame):
        """Detect a close forklift moving against the AMR's route heading."""
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None or self.current_waypoint_idx >= len(path.waypoints):
            return False
        tx, ty = path.waypoints[self.current_waypoint_idx]
        hx, hy = tx - self.agent_x, ty - self.agent_y
        heading_norm = math.hypot(hx, hy)
        if heading_norm < 1e-6:
            return False
        hx, hy = hx / heading_norm, hy / heading_norm
        for obj in frame["objects"]:
            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue
            cx = sum(point[0] for point in polygon) / len(polygon)
            cy = sum(point[1] for point in polygon) / len(polygon)
            dx, dy = cx - self.agent_x, cy - self.agent_y
            forward = dx * hx + dy * hy
            if not 0 < forward <= 120:
                continue
            vx, vy = self.observed_pixel_velocity(obj)
            velocity_norm = math.hypot(vx, vy)
            if velocity_norm < 5.0:
                continue
            # Dot <= -0.5 means the forklift approaches from the opposite
            # direction within 60 degrees of a direct head-on encounter.
            if (vx / velocity_norm) * hx + (vy / velocity_norm) * hy <= -0.5:
                return True
        return False

    def begin_safe_parking(self):
        if self.safe_parking_active or len(ROBOT_OPERATING_AREA_POINTS) < 3:
            return
        operating_area = Polygon(ROBOT_OPERATING_AREA_POINTS)
        _, border = nearest_points(Point(self.agent_x, self.agent_y), operating_area.boundary)
        self.safe_parking_target = (float(border.x), float(border.y))
        self.safe_parking_active = True
        self.safe_parking_arrived = Point(self.agent_x, self.agent_y).distance(border) <= 3.0
        self.safe_parking_clear_frames = 0

    def update_safe_parking(self):
        """Drive directly to the nearest area boundary, then hold safely."""
        if not self.safe_parking_active or self.safe_parking_target is None:
            return False
        if self.safe_parking_arrived:
            return True
        tx, ty = self.safe_parking_target
        dx, dy = tx - self.agent_x, ty - self.agent_y
        distance = math.hypot(dx, dy)
        if distance <= 3.0:
            self.agent_x, self.agent_y = tx, ty
            self.safe_parking_arrived = True
            return True
        path = DEFAULT_PATHS.get(self.current_path_name)
        speed = path.speed * self.agent_speed_factor if path else 2.0
        step = min(speed, distance)
        self.agent_x += dx / distance * step
        self.agent_y += dy / distance * step
        return True

    def resume_from_safe_parking(self):
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path and path.waypoints:
            self.current_waypoint_idx = min(
                range(len(path.waypoints)),
                key=lambda index: math.hypot(
                    path.waypoints[index][0] - self.agent_x,
                    path.waypoints[index][1] - self.agent_y,
                ),
            )
            if self.current_waypoint_idx == 0:
                self.current_waypoint_idx = 1
        self.safe_parking_active = False
        self.safe_parking_arrived = False
        self.safe_parking_target = None
        self.safe_parking_clear_frames = 0

    def update_agent(self):

        if self.dragging:
            return

        if self.update_safe_parking():
            return

        # STOP is immediate. A YIELD instead lets the AMR close up to the
        # intersection boundary, then holds there rather than stopping far
        # upstream when a forklift is approaching the intersection.
        if self.decision_action == "STOP":
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

        if self.decision_action == "SLOW":
            speed *= 0.5

        if self.decision_action == "CLEAR_INTERSECTION":
            speed *= 1.5

        if self.decision_action == "YIELD" and len(INTERSECTION_POINTS) >= 3:
            distance_to_entry = Point(self.agent_x, self.agent_y).distance(
                Polygon(INTERSECTION_POINTS)
            )
            if distance_to_entry <= 3.0:
                return
            speed = min(speed, max(0.0, distance_to_entry - 3.0))

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

    def reset_parallel_agents(self):
        """Reset one independent supervisor/robot state per selected route."""
        self.parallel_robot_states = []
        for replay_path in SCENARIO_ALL_PATHS:
            waypoints = [tuple(point) for point in replay_path.get("robot_path", [])]
            if not waypoints:
                continue
            self.parallel_robot_states.append({
                "replay": replay_path,
                "waypoints": waypoints,
                "speed": float(replay_path.get("robot_speed_px_per_frame", 2.0)),
                "x": float(waypoints[0][0]),
                "y": float(waypoints[0][1]),
                "waypoint_idx": 1,
                "decision_action": "GO",
                "completed": False,
            })
        self.parallel_last_frame = None

    def update_parallel_agents(self, target_frame):
        """Run each displayed route through the same per-frame supervisor loop.

        The viewer's normal single-robot update uses ``update_agent`` followed
        by ``evaluate_scene_risk``.  This method temporarily installs each
        independent state and calls those exact methods, so parallel markers
        yield, stop and clear just as they do when selected individually.
        """
        if not SCENARIO_ALL_PATHS:
            return
        if not self.parallel_robot_states:
            self.reset_parallel_agents()
        # Cached behavior makes parallel playback a drawing operation rather
        # than re-running ten supervisor simulations on every GUI frame.
        if self.use_precomputed_behavior and all(
            state["replay"].get("precomputed_behavior", {}).get("positions")
            for state in self.parallel_robot_states
        ):
            for state in self.parallel_robot_states:
                cached = state["replay"]["precomputed_behavior"]
                # All paths in a window share their start frame but can have
                # different end frames.  Once a shorter replay reaches its
                # final cached state, hold that AMR at its goal instead of
                # falling back to live simulation (which starts it again).
                cache_index = min(target_frame, len(cached["positions"]) - 1)
                state["x"], state["y"] = cached["positions"][cache_index]
                state["decision_action"] = cached["actions"][cache_index]
                state["completed"] = cache_index == len(cached["positions"]) - 1
            self.parallel_last_frame = target_frame
            return
        if self.parallel_last_frame is None or target_frame != self.parallel_last_frame + 1:
            self.reset_parallel_agents()
            frame_range = range(0, target_frame + 1)
        else:
            frame_range = (target_frame,)

        saved = (
            self.agent_x, self.agent_y, self.current_waypoint_idx,
            self.current_path_name, self.decision_action, self.frame_idx,
        )
        temporary_name = "__parallel_replay_path__"
        temporary_path = RobotPath(temporary_name)
        DEFAULT_PATHS[temporary_name] = temporary_path
        try:
            for frame_index in frame_range:
                self.frame_idx = frame_index
                source_frame = tracks[frame_index]
                self.current_timestamp = source_frame["timestamp"]
                for state in self.parallel_robot_states:
                    # Unlike an editable live path, a replay route has a
                    # terminal goal.  Do not let ``update_agent`` interpret
                    # an exhausted waypoint index as a request to snap back
                    # to the route start.
                    if state["completed"]:
                        continue
                    temporary_path.waypoints = state["waypoints"]
                    temporary_path.speed = state["speed"]
                    self.current_path_name = temporary_name
                    self.agent_x, self.agent_y = state["x"], state["y"]
                    self.current_waypoint_idx = state["waypoint_idx"]
                    self.decision_action = state["decision_action"]
                    self.update_agent()
                    state["x"], state["y"] = self.agent_x, self.agent_y
                    state["waypoint_idx"] = self.current_waypoint_idx
                    self.evaluate_scene_risk(source_frame)
                    state["decision_action"] = self.decision_action
                    if state["waypoint_idx"] >= len(state["waypoints"]):
                        state["completed"] = True
        finally:
            DEFAULT_PATHS.pop(temporary_name, None)
            (
                self.agent_x, self.agent_y, self.current_waypoint_idx,
                self.current_path_name, self.decision_action, self.frame_idx,
            ) = saved
        self.parallel_last_frame = target_frame
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

        cached_behavior = SCENARIO_REPLAY.get("precomputed_behavior", {}) if SCENARIO_REPLAY else {}
        if self.use_precomputed_behavior and len(cached_behavior.get("positions", [])) > idx:
            self.agent_x, self.agent_y = cached_behavior["positions"][idx]
            self.decision_action = cached_behavior["actions"][idx]
        else:
            self.update_agent()
        self.update_parallel_agents(idx)

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

        self.draw_reference_paths()

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

        # A replay may explicitly contain an OOD override. Recorded-data
        # replays omit it and therefore retain the original OOD measurement.
        replay_frame = SCENARIO_FRAME_META.get(int(self.current_timestamp))
        if replay_frame is not None and "ood" in replay_frame:
            replay_ood = replay_frame.get("ood", {})
            self.alert_ood = (
                float(replay_ood.get("score", 0.0)),
                bool(replay_ood.get("flag", False)),
            )

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
        self.replay_finished = False

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
        self.scenario_select = None
        self.scenario_window_select = None
        self.scenario_path_select = None
        self.scenario_catalog = {}
        self.load_scenario_btn = None
        if SCENARIO_REPLAY_FILE:
            instances = load_json_file(repo_path(SCENARIO_REPLAY_FILE)).get("instances", [])
            for instance in instances:
                window_id = instance.get("window_id", instance.get("id"))
                self.scenario_catalog.setdefault(window_id, []).append(instance)
            self.scenario_window_select = QComboBox()
            self.scenario_path_select = QComboBox()
            for window_id in sorted(self.scenario_catalog):
                first = self.scenario_catalog[window_id][0]
                label = "{} — forklift {}".format(
                    window_id, first.get("focus_track_id", "?")
                )
                self.scenario_window_select.addItem(label, window_id)
            active_window = SCENARIO_REPLAY.get("window_id", SCENARIO_REPLAY.get("id"))
            selected_index = self.scenario_window_select.findData(active_window)
            if selected_index >= 0:
                self.scenario_window_select.setCurrentIndex(selected_index)
            self.populate_scenario_paths()
            self.load_scenario_btn = QPushButton("Load Scenario")
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
        self.precomputed_playback_check = QCheckBox("Use precomputed replay behavior")
        self.precomputed_playback_check.setChecked(True)

        self.show_reference_paths_check = QCheckBox("Show reference paths")
        self.reference_path_select = QComboBox()
        self.new_reference_path_btn = QPushButton("New Reference Path")
        self.edit_reference_path_btn = QPushButton("Edit Reference Path")
        self.save_reference_path_btn = QPushButton("Save Reference Path")
        self.refresh_reference_path_combo()

       

        ctrl = QVBoxLayout()

        # ==================================================
        # Playback
        # ==================================================

        ctrl.addWidget(self.load_btn)
        ctrl.addWidget(self.play_btn)
        ctrl.addWidget(self.reset_btn)
        ctrl.addWidget(self.label_frame)
        ctrl.addWidget(self.slider)
        ctrl.addWidget(self.precomputed_playback_check)

        if self.scenario_window_select is not None:
            ctrl.addWidget(QLabel("Recorded Time Window"))
            ctrl.addWidget(self.scenario_window_select)
            ctrl.addWidget(QLabel("Robot Path"))
            ctrl.addWidget(self.scenario_path_select)
            ctrl.addWidget(self.load_scenario_btn)

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

        ctrl.addWidget(QLabel("Robot Reference Paths"))
        ctrl.addWidget(self.show_reference_paths_check)
        ctrl.addWidget(self.reference_path_select)
        ctrl.addWidget(self.new_reference_path_btn)
        ctrl.addWidget(self.edit_reference_path_btn)
        ctrl.addWidget(self.save_reference_path_btn)

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
        self.precomputed_playback_check.toggled.connect(self.toggle_precomputed_playback)
        self.object_select.currentTextChanged.connect(self.change_object)        
        self.path_select.currentTextChanged.connect(self.change_path)
        self.edit_path_btn.clicked.connect(self.toggle_path_edit)
        self.new_path_btn.clicked.connect(self.create_new_path)
        self.save_path_btn.clicked.connect(self.save_path_library)
        self.load_path_btn.clicked.connect(self.load_path_library)
        self.show_reference_paths_check.toggled.connect(self.toggle_reference_paths)
        self.new_reference_path_btn.clicked.connect(self.create_reference_path)
        self.edit_reference_path_btn.clicked.connect(self.edit_reference_path)
        self.save_reference_path_btn.clicked.connect(self.save_reference_path)
        if self.load_scenario_btn is not None:
            self.load_scenario_btn.clicked.connect(self.load_selected_scenario)
            self.scenario_window_select.currentIndexChanged.connect(self.populate_scenario_paths)

    def toggle_precomputed_playback(self, checked):
        self.viewer.use_precomputed_behavior = checked
        self.viewer.reset_parallel_agents()
        self.viewer.update_frame(self.viewer.frame_idx)

    def load_selected_scenario(self):
        window_id = self.scenario_window_select.currentData()
        path_value = self.scenario_path_select.currentData()
        if not window_id or not path_value:
            return
        show_all_paths = path_value == "__all__"
        if show_all_paths:
            scenario_id = self.scenario_catalog[window_id][0].get("id")
        else:
            scenario_id = path_value
        self.timer.stop()
        activate_scenario_replay(str(scenario_id), show_all_paths=show_all_paths)
        self.replay_finished = False
        self.play_btn.setText("▶ Play")
        self.viewer.current_path_name = "Path_1"
        self.viewer.active_track_id = int(SCENARIO_REPLAY["focus_track_id"])
        self.viewer.frame_idx = 0
        self.viewer.decision_action = "GO"
        self.viewer.intersection_reason = None
        self.viewer.safe_parking_active = False
        self.viewer.safe_parking_arrived = False
        self.viewer.safe_parking_target = None
        self.viewer.safe_parking_clear_frames = 0
        self.viewer.snap_agent_to_active_path()
        self.viewer.reset_parallel_agents()
        self.slider.blockSignals(True)
        self.slider.setRange(0, NUM_FRAMES - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.viewer.update_frame(0)
        self.refresh_object_combo()
        self.refresh_info_panel()
        self.label_frame.setText(
            "Frame: 0 | TS: {}".format(self.viewer.current_timestamp)
        )
        self.statusBar().showMessage(
            "Loaded {}{}".format(
                window_id, " — all robot paths" if show_all_paths else ""
            )
        )

    def populate_scenario_paths(self):
        """Populate the route selector for the currently selected recording."""
        if self.scenario_window_select is None:
            return
        window_id = self.scenario_window_select.currentData()
        variants = self.scenario_catalog.get(window_id, [])
        self.scenario_path_select.blockSignals(True)
        self.scenario_path_select.clear()
        if variants:
            self.scenario_path_select.addItem("All 10 robot paths (parallel)", "__all__")
            for instance in variants:
                self.scenario_path_select.addItem(
                    instance.get("robot_path_label", instance.get("id", "Robot path")),
                    instance.get("id"),
                )
        self.scenario_path_select.blockSignals(False)

    def refresh_reference_path_combo(self, selected_name=None):
        reference_paths = load_reference_paths()
        self.reference_path_select.blockSignals(True)
        self.reference_path_select.clear()
        for name in sorted(reference_paths, key=reference_path_sort_key):
            self.reference_path_select.addItem(name)
        if selected_name:
            index = self.reference_path_select.findText(selected_name)
            if index >= 0:
                self.reference_path_select.setCurrentIndex(index)
        self.reference_path_select.blockSignals(False)

    def toggle_reference_paths(self, checked):
        self.viewer.show_reference_paths = checked
        self.viewer.update_frame(self.viewer.frame_idx)

    def create_reference_path(self):
        reference_paths = load_reference_paths()
        number = 1
        while "Path_{}".format(number) in reference_paths:
            number += 1
        name = "Path_{}".format(number)
        self.viewer.reference_path_edit_name = name
        self.viewer.reference_path_edit_waypoints = []
        self.viewer.reference_path_edit_mode = True
        self.viewer.show_reference_paths = True
        self.show_reference_paths_check.setChecked(True)
        self.refresh_reference_path_combo(name)
        self.edit_reference_path_btn.setText("Editing Reference Path")
        self.statusBar().showMessage(
            "Drawing {}: left-click adds a waypoint; right-click removes the nearest one.".format(name)
        )
        self.viewer.update_frame(self.viewer.frame_idx)

    def edit_reference_path(self):
        name = self.reference_path_select.currentText()
        reference_paths = load_reference_paths()
        definition = reference_paths.get(name)
        if not definition:
            return
        self.viewer.reference_path_edit_name = name
        self.viewer.reference_path_edit_waypoints = [
            [float(x), float(y)] for x, y in definition.get("waypoints", [])
        ]
        self.viewer.reference_path_edit_mode = True
        self.viewer.show_reference_paths = True
        self.show_reference_paths_check.setChecked(True)
        self.edit_reference_path_btn.setText("Editing Reference Path")
        self.statusBar().showMessage(
            "Editing {}: left-click adds a waypoint; right-click removes the nearest one.".format(name)
        )
        self.viewer.update_frame(self.viewer.frame_idx)

    def save_reference_path(self):
        name = self.viewer.reference_path_edit_name
        waypoints = self.viewer.reference_path_edit_waypoints
        if not name or len(waypoints) < 2:
            self.statusBar().showMessage("A reference path needs at least two waypoints before saving.")
            return
        reference_paths = load_reference_paths()
        existing = reference_paths.get(name, {})
        reference_paths[name] = {
            "speed": float(existing.get("speed", 2.0)),
            "start_frame": existing.get("start_frame", 0),
            "end_frame": existing.get("end_frame"),
            "associated_track": existing.get("associated_track"),
            "waypoints": waypoints,
        }
        save_reference_paths(reference_paths)
        self.viewer.reference_path_edit_mode = False
        self.refresh_reference_path_combo(name)
        self.edit_reference_path_btn.setText("Edit Reference Path")
        self.statusBar().showMessage("Saved {} to {}".format(name, REFERENCE_PATHS_PATH.name))
        self.viewer.update_frame(self.viewer.frame_idx)

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

        # Unlike the full recording, an instance replay is finite. Freeze on
        # its last frame rather than wrapping into an invalid source index.
        if SCENARIO_REPLAY and self.viewer.frame_idx >= NUM_FRAMES - 1:
            self.timer.stop()
            self.replay_finished = True
            self.play_btn.setText("↻ Replay")
            self.statusBar().showMessage(
                "Replay complete — press Replay or Reset to start from frame 0."
            )
            return

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

        replay_focus_id = (
            int(SCENARIO_REPLAY["focus_track_id"])
            if SCENARIO_REPLAY and SCENARIO_REPLAY.get("focus_track_id") is not None
            else None
        )
        if replay_focus_id is not None:
            # The focus track may be absent in a few early recorded frames.
            # Retain its identity and select it automatically as soon as its
            # recorded detection appears; never substitute an unrelated track.
            selected = replay_focus_id
            self.viewer.active_track_id = replay_focus_id

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

        elif replay_focus_id is not None:

            self.object_select.setCurrentIndex(-1)

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
            if SCENARIO_REPLAY and (
                self.replay_finished or self.viewer.frame_idx >= NUM_FRAMES - 1
            ):
                self.reset()
            self.replay_finished = False
            self.timer.start()
            self.play_btn.setText("⏸ Pause")

    def reset(self):
        self.timer.stop()
        self.replay_finished = False
        self.play_btn.setText("▶ Play")
        path = DEFAULT_PATHS[
            self.viewer.current_path_name
        ]
        if len(path.waypoints):
            self.viewer.agent_x = path.waypoints[0][0]
            self.viewer.agent_y = path.waypoints[0][1]
            self.viewer.current_waypoint_idx = 1
        self.viewer.decision_action = "GO"
        self.viewer.intersection_reason = None
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
