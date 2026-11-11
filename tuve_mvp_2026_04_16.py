# ==========================================================
# TUVE MVP FULL PyQt6 Port (Feature Complete + Info Panel)
# ==========================================================

import sys, os, json, cv2, pickle 
import numpy as np
import math
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel,
    QComboBox
)
from collections import deque

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from trajectory_player import MLModel  
from safety_framework import (
    SceneContext,
    Supervisor,
    SupervisorManager,
    SupervisorResult,
    BBoxUncertaintySupervisor,
    TrajectoryUncertaintySupervisor,
    RouteUncertaintySupervisor,
    ManeuverUncertaintySupervisor,
    DecisionEngine
)
from dataclasses import dataclass, field

# ==========================================================
# CONFIG
# ==========================================================
CONFIG_FILE = "mvp-config.json"

with open(CONFIG_FILE, "r") as _f:
    CFG = json.load(_f)

DATA_ROOT = Path(CFG["paths"]["dataset_root"])
EXTRA_PATHS = CFG["extra_paths"]

MASK_FOLDER = str(DATA_ROOT / EXTRA_PATHS["mask_folder"])
SAVE_PATH = str(DATA_ROOT / CFG["paths"]["save_path"])
PRED_PATH = str(DATA_ROOT / EXTRA_PATHS["predictions"])
OOD_ROOT = str(DATA_ROOT / CFG["paths"]["ood_root"])

CAMERA_LOOKUP_PATH = str(
    DATA_ROOT / CFG["paths"]["camera_lookup_table"]
)

OCCUPANCY_PATH = str(
    DATA_ROOT / EXTRA_PATHS["occupancy_projection"]
)

PATH_DB = str(
    DATA_ROOT / EXTRA_PATHS["robot_paths"]
)

MODEL_PATH = str(
    EXTRA_PATHS["model_path"]
)

VEHICLE_RADIUS = CFG["runtime"]["vehicle_radius"]
MAX_DISTANCE = CFG["runtime"]["max_distance"]

OOD_CACHE = {}

MODEL = MLModel.load(MODEL_PATH)
ENTRY_ELLIPSES = MODEL.entry_ellipses
INTERSECTION_POLY = getattr(
    MODEL,
    "intersection_poly",
    None
)

# ==========================================================
# LOAD STATIC DATA
# ==========================================================
CACHE_FILE = "tuve_cache.pkl"



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

def build_track_cache(tracks):

    track_unc = {}
    track_hist = {}
    bbox_stats = {}

    for frame_idx, frame in enumerate(tracks):

        ts = frame["timestamp"]

        for obj in frame["objects"]:

            tid = obj["track_id"]

            wx, wy = obj["center_world"]
            cx, cy = world_to_occupancy(wx, wy)
            obj["occ_center"] = (cx, cy)

            track_hist.setdefault(
                tid,
                []
            ).append(
                (
                    frame_idx,
                    cx,
                    cy
                )
            )

            poly = np.asarray(
                obj["polygon"],
                dtype=float
            )

            if len(poly) < 3:
                continue

            center = poly.mean(0)

            track_unc.setdefault(
                tid,
                []
            ).append({
                "center": center,
                "area": cv2.contourArea(poly.astype(np.float32)),
                "frame": ts
            })

            ang = np.deg2rad(obj["angle"])

            Rm = np.array([
                [np.cos(ang), np.sin(ang)],
                [-np.sin(ang), np.cos(ang)]
            ])

            local = (poly-center) @ Rm.T

            bbox_stats.setdefault(
                tid,
                []
            ).append(
                (
                    max(
                        0.0,
                        (
                            np.ptp(local[:,0])
                            -
                            float(obj["bbox_length"])
                        )
                    ),
                    max(
                        0.0,
                        (
                            np.ptp(local[:,1])
                            -
                            float(obj["bbox_width"])
                        )
                    )
                )
            )

    return (
        track_unc,
        track_hist,
        bbox_stats
    )

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

            poly = obj.get("associated_polygon", [])

            frame_objects.append({
                "track_id": track_id,
                "timestamp": timestamp,

                "center_world": [x, y],
                "velocity_world": [vx, vy],

                "bbox_width": float(width),
                "bbox_length": float(length),

                "polygon": poly,

                "angle": float(angle)
            })

        tracks.append({
            "timestamp": timestamp,
            "objects": frame_objects
        })

    return tracks

def estimate_track_uncertainty(
    frame_idx,
    track_id
):

    hist = TRACK_UNCERTAINTY.get(
        track_id
    )

    if not hist or len(hist) < 3:
        return 0.0

    recent = hist[-5:]

    centers = np.asarray(
        [r["center"] for r in recent]
    )

    areas = np.asarray(
        [r["area"] for r in recent]
    )

    return float(
        np.clip(
            0.7 *
            min(
                1.0,
                np.linalg.norm(
                    centers.std(0)
                ) / 15
            )
            +
            0.3 *
            min(
                1.0,
                areas.std()
                /
                (
                    areas.mean()+1e-6
                )
            ),
            0,
            1
        )
    )
# ==========================================================
# HELPERS 
# ==========================================================

def get_occ_center(obj):
    cached = obj.get("occ_center")
    if cached is not None:
        return cached

    wx, wy = obj["center_world"]
    cached = world_to_occupancy(wx, wy)
    obj["occ_center"] = cached
    return cached

def world_to_occupancy(wx, wy):

    # assumption A:
    X = np.array([
        [wx],
        [wy],
        [0.0],
        [1.0]
    ], dtype=np.float64)

    uvw = P @ X
    uvw = uvw.ravel()

    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]

    return float(u), float(v)

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

    cam_id = str(cam_id)

    cached = OOD_CACHE.get(cam_id)

    if cached is not None:
        return cached

    npz = (
        Path(OOD_ROOT)
        / cam_id
        / "ood_score.npz"
    )

    if not npz.exists():

        result = (
            {},
            np.empty(
                0,
                dtype=np.int64
            )
        )

        OOD_CACHE[cam_id] = result

        return result

    dtype = np.dtype([
        ("id", np.uint64),
        ("ood_score", np.float32),
        ("pred", np.uint8)
    ])

    arr = np.fromfile(
        npz,
        dtype=dtype
    )

    scores = {
        int(i): (
            float(s),
            bool(p)
        )
        for i, s, p in arr
    }

    keys = arr["id"].astype(
        np.int64
    )

    result = (
        scores,
        keys
    )

    OOD_CACHE[cam_id] = result

    return result

# ==========================================================
# DATA LOAD
# ==========================================================

if Path(CACHE_FILE).exists():

    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)

    tracks = cache["tracks"]
    TRACK_HISTORY = cache["history"]
    TRACK_UNCERTAINTY = cache["unc"]
    TRACK_BBOX_STATS = cache["bbox"]

else:

    tracks, uncertainties, data_mode = (
        load_tracks_auto(SAVE_PATH)
    )

    (
        TRACK_UNCERTAINTY,
        TRACK_HISTORY,
        TRACK_BBOX_STATS
    ) = build_track_cache(tracks)

    with open(CACHE_FILE, "wb") as f:

        pickle.dump(
            {
                "tracks": tracks,
                "history": TRACK_HISTORY,
                "unc": TRACK_UNCERTAINTY,
                "bbox": TRACK_BBOX_STATS
            },
            f
        )

TRACK_HISTORY_IDX = {
    tid: np.asarray(
        [h[0] for h in hist],
        dtype=np.int32
    )
    for tid, hist in TRACK_HISTORY.items()
}

first_ts = tracks[0]["timestamp"]

mask_file = mask_lookup.get(first_ts)

if mask_file is None:

    raise RuntimeError(
        f"No BEV image for timestamp {first_ts}"
    )

mask0 = cv2.imread(
    os.path.join(
        MASK_FOLDER,
        mask_file
    ),
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
# BBOX UNCERTAINTY MODEL
# ==========================================================

def uncertainty_bbox(track_id):

    stats = TRACK_BBOX_STATS.get(
        track_id
    )

    if not stats:
        return (0.0, 0.0)

    arr = np.asarray(
        stats,
        dtype=float
    )

    expand_l = np.percentile(
        arr[:, 0],
        95
    )

    expand_w = np.percentile(
        arr[:, 1],
        95
    )

    return (
        float(expand_l),
        float(expand_w)
    )

TRACK_BBOX_MODEL = {}

for tid, samples in TRACK_BBOX_STATS.items():

    arr = np.asarray(
        samples,
        dtype=float
    )

    if len(arr) < 5:
        continue

    TRACK_BBOX_MODEL[tid] = {

        "mean":
            arr.mean(0),

        "cov":
            np.cov(arr.T)
    }

def estimate_bbox_uncertainty(
    track_id
):

    model = (
        TRACK_BBOX_MODEL.get(
            track_id
        )
    )

    if model is None:
        return 0.0

    eigvals = np.linalg.eigvalsh(
        model["cov"]
    )

    raw = np.sqrt(
        eigvals.max()
    )

    return float(
        np.clip(
            raw / 2.0,
            0.0,
            1.0
        )
    )

def estimate_trajectory_uncertainty(
    track_id
):

    hist = TRACK_HISTORY.get(
        track_id,
        []
    )

    if len(hist) < 6:
        return 0.0

    pts = np.asarray([

        (h[1], h[2])

        for h in hist[-10:]
    ])

    vel = np.diff(
        pts,
        axis=0
    )

    if len(vel) < 2:
        return 0.0

    headings = np.arctan2(
        vel[:,1],
        vel[:,0]
    )

    return float(

        np.clip(
            np.std(headings)
            /
            np.pi,
            0.0,
            1.0
        )
    )

def route_prediction_uncertainty(
    route_info
):

    if route_info is None:
        return 1.0

    p = route_info.get(
        "prob"
    )

    if p is None:
        return 1.0

    return float(
        np.clip(
            1.0-float(p),
            0.0,
            1.0
        )
    )

def maneuver_uncertainty(
    mode_info
):

    if mode_info is None:
        return 1.0

    conf = mode_info.get(
        "display_confidence"
    )

    if conf is None:
        return 1.0

    return float(
        np.clip(
            1.0-float(conf),
            0.0,
            1.0
        )
    )

@dataclass
class ScenarioResult:

    name: str

    score: float

    flag: bool

    details: dict = field(
        default_factory=dict
    )

class ScenarioEngine:

    def __init__(self):

        self.scenarios = []

    def register(self, scenario):

        self.scenarios.append(
            scenario
        )

    def evaluate(self, scene):

        return [
            s.evaluate(scene)
            for s in self.scenarios
        ]
    
class OODSupervisor(Supervisor):
    name = "camera_ood"
    category = "input_anomaly"

    def evaluate(
        self,
        scene
    ):

        score = 0.0
        flag = False

        alert = scene.extra.get(
            "alert_ood"
        )

        if alert is not None:

            score = float(
                alert[0]
            )

            flag = bool(
                alert[1]
            )

        return SupervisorResult(

            name=self.name,

            category=self.category,

            score=score,

            confidence=max(
                0.0,
                1.0 - score
            ),

            flag=flag,

            details={
                "ood_score": score,
                "ood_flag": flag
            }
        )

class LegacyTrackUncertaintySupervisor(
    Supervisor
):

    name = "track_uncertainty"

    category = "uncertainty"

    def evaluate(
        self,
        scene
    ):

        score = 0.0

        if (
            scene.active_track_id
            is not None
        ):

            score = (
                estimate_track_uncertainty(
                    scene.frame_idx,
                    scene.active_track_id
                )
            )

        return SupervisorResult(

            name=self.name,

            category=self.category,

            score=score,

            confidence=
                max(
                    0.0,
                    1.0-score
                ),

            flag=(score > 0.7)
        )

class IntersectionOccupancyScenario:

    name = "intersection_occupancy"
    category = "scenario"

    OCCUPANCY_TIME = 3.0

    @staticmethod
    def point_in_poly(point, poly):

        x, y = point

        inside = False
        j = len(poly) - 1

        for i in range(len(poly)):

            xi, yi = poly[i]
            xj, yj = poly[j]

            if (
                ((yi > y) != (yj > y))
                and
                (
                    x <
                    (xj - xi)
                    * (y - yi)
                    /
                    (yj - yi + 1e-12)
                    + xi
                )
            ):
                inside = not inside

            j = i

        return inside

    @staticmethod
    def poly_center(poly):

        return np.asarray(
            poly,
            dtype=float
        ).mean(axis=0)

    @classmethod
    def distance_to_intersection_along_path(
        cls,
        viewer,
        poly_occ
    ):

        path = DEFAULT_PATHS.get(
            viewer.current_path_name
        )

        if (
            path is None
            or
            viewer.current_waypoint_idx >=
            len(path.waypoints)
        ):
            return np.inf

        route = [
            (
                viewer.agent_x,
                viewer.agent_y
            )
        ]

        route.extend(
            path.waypoints[
                viewer.current_waypoint_idx:
            ]
        )

        travelled = 0.0

        for i in range(len(route) - 1):

            p0 = np.asarray(
                route[i],
                dtype=float
            )

            p1 = np.asarray(
                route[i + 1],
                dtype=float
            )

            seg = p1 - p0

            seg_len = np.linalg.norm(seg)

            if seg_len < 1e-6:
                continue

            steps = max(
                2,
                int(seg_len / 25)
            )

            for k in range(steps + 1):

                p = (
                    p0
                    + seg * k / steps
                )

                if cls.point_in_poly(
                    p,
                    poly_occ
                ):
                    return travelled

                travelled += (
                    seg_len / steps
                )

        return np.inf

    def evaluate(self, scene):

        viewer = scene.extra.get(
            "viewer"
        )

        poly_world = scene.extra.get(
            "intersection_poly"
        )

        if (
            viewer is None
            or poly_world is None
        ):
            return ScenarioResult(
                name=self.name,
                score=0.0,
                flag=False,
                details={}
            )

        poly_occ = world_poly_to_occ(
            poly_world
        )

        if (
            poly_occ is None
            or len(poly_occ) < 3
        ):
            return ScenarioResult(
                name=self.name,
                score=0.0,
                flag=False,
                details={}
            )

        center = self.poly_center(
            poly_occ
        )

        #
        # AMR ETA
        #

        robot_distance = (
            self.distance_to_intersection_along_path(
                viewer,
                poly_occ
            )
        )

        if not np.isfinite(
            robot_distance
        ):
            return ScenarioResult(
                name=self.name,
                score=0.0,
                flag=False,
                details={}
            )

        robot_speed = max(
            0.5,
            viewer.agent_speed_factor
        )

        robot_eta = (
            robot_distance /
            robot_speed
        )

        #
        # only care when AMR is nearing
        # intersection
        #

        if robot_eta > 5.0:

            return ScenarioResult(
                name=self.name,
                score=0.0,
                flag=False,
                details={
                    "robot_eta":
                        float(robot_eta)
                }
            )

        best_risk = 0.0
        conflict_track = None

        active_track = getattr(
            viewer,
            "active_track_id",
            None
        )

        for obj in scene.frame["objects"]:

            tid = obj["track_id"]

            if (
                active_track is not None
                and tid != active_track
            ):
                continue

            cx, cy = get_occ_center(obj)

            vx, vy = obj["velocity_world"]

            speed = np.hypot(
                vx,
                vy
            )

            #
            # stationary forklift
            #

            if speed < 0.2:
                continue

            #
            # moving toward intersection?
            #

            to_intersection = (
                center
                -
                np.array(
                    [cx, cy]
                )
            )

            if np.dot(
                to_intersection,
                np.array([vx, vy])
            ) <= 0:
                continue

            forklift_distance = (
                np.linalg.norm(
                    to_intersection
                )
            )

            forklift_eta = (
                forklift_distance
                /
                speed
            )

            risk = np.clip(
                1.0
                -
                (
                    abs(
                        robot_eta
                        -
                        forklift_eta
                    )
                    /
                    self.OCCUPANCY_TIME
                ),
                0.0,
                1.0
            )

            if risk > best_risk:

                best_risk = risk
                conflict_track = tid

        return ScenarioResult(

            name=self.name,

            score=float(
                best_risk
            ),

            flag=(
                best_risk > 0.5
            ),

            details={

                "robot_eta":
                    float(robot_eta),

                "occupancy_risk":
                    float(best_risk),

                "conflicting_track":
                    conflict_track
            }
        )
        
class TrajectoryConflictScenario:

    name = "trajectory_conflict"
    category = "scenario"

    SAFE_DISTANCE = 80.0
    CRITICAL_DISTANCE = 40.0

    LOOKAHEAD_POINTS = 10

    def evaluate(self, scene):

        viewer = scene.extra.get(
            "viewer"
        )

        if viewer is None:

            return ScenarioResult(
                name=self.name,
                score=0.0,
                flag=False,
                details={}
            )

        path = DEFAULT_PATHS.get(
            viewer.current_path_name
        )

        if (
            path is None
            or
            viewer.current_waypoint_idx >=
            len(path.waypoints)
        ):
            return ScenarioResult(
                name=self.name,
                score=0.0,
                flag=False,
                details={}
            )

        #
        # Future AMR route points
        #

        future_points = [
            np.array(
                [
                    viewer.agent_x,
                    viewer.agent_y
                ],
                dtype=float
            )
        ]

        end_idx = min(
            len(path.waypoints),
            viewer.current_waypoint_idx
            +
            self.LOOKAHEAD_POINTS
        )

        for i in range(
            viewer.current_waypoint_idx,
            end_idx
        ):

            future_points.append(
                np.array(
                    path.waypoints[i],
                    dtype=float
                )
            )

        highest_risk = 0.0

        conflict_track = None

        conflict_distance = np.inf

        for obj in scene.frame["objects"]:

            tid = obj["track_id"]

            cx, cy = get_occ_center(obj)

            obj_pos = np.array(
                [cx, cy],
                dtype=float
            )

            for p in future_points:

                d = np.linalg.norm(
                    p - obj_pos
                )

                if d >= self.SAFE_DISTANCE:
                    continue

                risk = np.clip(
                    (
                        self.SAFE_DISTANCE
                        - d
                    )
                    /
                    (
                        self.SAFE_DISTANCE
                        - self.CRITICAL_DISTANCE
                    ),
                    0.0,
                    1.0
                )

                if risk > highest_risk:

                    highest_risk = risk
                    conflict_track = tid
                    conflict_distance = d

        return ScenarioResult(

            name=self.name,

            score=float(
                highest_risk
            ),

            flag=(
                highest_risk > 0.5
            ),

            details={

                "conflicting_track":
                    conflict_track,

                "minimum_distance":
                    float(
                        conflict_distance
                    ),

                "avoid_side":
                    "LEFT"
            }
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


def make_rotated_bbox(
    cx,
    cy,
    width,
    length,
    angle_deg
):

    hw = width * 0.5
    hl = length * 0.5

    pts = np.array([
        [-hl, -hw],
        [ hl, -hw],
        [ hl,  hw],
        [-hl,  hw],
        [-hl, -hw]
    ])

    ang = np.deg2rad(angle_deg)

    R = np.array([
        [np.cos(ang), -np.sin(ang)],
        [np.sin(ang),  np.cos(ang)]
    ])

    pts = pts @ R.T

    pts[:,0] += cx
    pts[:,1] += cy

    return pts
def world_poly_to_occ(poly):

    if poly is None:
        return None

    out = []

    for x, y in np.asarray(poly):

        u, v = world_to_occupancy(
            float(x),
            float(y)
        )

        out.append([u, v])

    return np.asarray(
        out,
        dtype=float
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
        self.entry_artists = []
        self.trajectory_artist = None
        self.object_artists = {}
        self.text_artists = {}
        self.velocity_artists = {}
        self.path_line = None
        self.path_points = None
        self.path_target = None        
        self.intersection_patch = None
        self.intersection_label = None

        self.track_history = {}        

        self.frame_idx = 0
        self.draw_entry_points()
        self.draw_intersection()

        self.alert_collision = False
        self.alert_cam = None
        self.active_track_id = None
        self.avoidance_active = False
        self.avoidance_side = None
        self.avoidance_offset = 40.0
        self.current_lane_offset = 0.0
        self.target_lane_offset = 0.0
        self.route_info = None
        self.mode_info = None
        self.operator_speed_factor = 1.0
        self.track_ids_current = []
        self.trajectory_artist = None        
        self.current_traj_len = 0
        self.minimum_ttc = 999.0
        self.current_trust = 1.0
        self.runtime_state = "NORMAL"
        self.decision_action = "GO"
        self.near_miss = False
        self.failed_test = False
        self.evidence_log = deque(maxlen=10000)
        self.supervisor_manager = (
            SupervisorManager()
        )

        self.supervisor_manager.register(
            OODSupervisor()
        )

        self.supervisor_manager.register(
            LegacyTrackUncertaintySupervisor()
        )

        self.supervisor_manager.register(
            BBoxUncertaintySupervisor()
        )

        self.supervisor_manager.register(
            TrajectoryUncertaintySupervisor()
        )

        self.supervisor_manager.register(
            RouteUncertaintySupervisor()
        )

        self.supervisor_manager.register(
            ManeuverUncertaintySupervisor()
        )
        self.scenario_engine = (
            ScenarioEngine()
        )

        self.scenario_engine.register(
            TrajectoryConflictScenario()
        )

        self.scenario_engine.register(
            IntersectionOccupancyScenario()
        )
        
        self.supervisor_results = []

        self.decision_engine = (
            DecisionEngine()
        )


        self.route_history = deque(
            maxlen=10
        )

        self.mode_history = deque(
            maxlen=10
        )

        self.route_consistency = 1.0
        self.mode_consistency = 1.0

        self.prediction_trust = 1.0

        # prediction lookup cache
        self._last_lookup_key = None


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

    def draw_intersection(self):

        if INTERSECTION_POLY is None:
            return

        poly = world_poly_to_occ(
            INTERSECTION_POLY
        )

        if poly is None:
            return

        color = "gold"

        risk = getattr(
            self,
            "intersection_risk",
            0.0
        )

        if risk > 0.50:
            color = "red"

        elif risk > 0.20:
            color = "orange"

        if self.intersection_patch is None:

            self.intersection_patch = self.ax.fill(
                poly[:,0],
                poly[:,1],
                color=color,
                alpha=0.20,
                zorder=6
            )[0]

            x = poly[:,0].mean()
            y = poly[:,1].mean()

            self.intersection_label = self.ax.text(
                x,
                y,
                "INTERSECTION",
                color="darkorange",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="black",
                    edgecolor="gold",
                    alpha=0.7
                ),
                zorder=7
            )

        else:

            self.intersection_patch.set_xy(poly)

            self.intersection_patch.set_color(
                color
            )

    def get_object_artist(self, tid):
        artist = self.object_artists.get(tid)

        if artist is not None:
            return artist

        outer, = self.ax.plot(
            [],
            [],
            color="orange",
            lw=1.5,
            alpha=0.6,
            zorder=25
        )

        inner, = self.ax.plot(
            [],
            [],
            color="cyan",
            lw=2,
            zorder=30
        )

        arrow, = self.ax.plot(
            [],
            [],
            color="yellow",
            lw=2,
            zorder=35
        )

        text = self.ax.text(
            0,
            0,
            "",
            color="yellow",
            fontsize=8,
            ha="center"
        )

        self.object_artists[tid] = (
            outer,
            inner,
            arrow,
            text
        )

        return self.object_artists[tid]

    def update_avoidance_state(
        self,
        conflict_risk,
        avoid_side="LEFT"
    ):

        ENTER_THRESHOLD = 0.75
        EXIT_THRESHOLD  = 0.40

        if (
            not self.avoidance_active
            and
            conflict_risk > ENTER_THRESHOLD
        ):

            self.avoidance_active = True
            self.avoidance_side = avoid_side

        elif (
            self.avoidance_active
            and
            conflict_risk < EXIT_THRESHOLD
        ):

            self.avoidance_active = False
            self.avoidance_side = None

        if self.avoidance_active:
            if self.avoidance_side == "LEFT":
                self.target_lane_offset = -self.avoidance_offset
            else:
                self.target_lane_offset = 0.0
        else:
            self.target_lane_offset = 0.0


    def get_offset_target(self, tx, ty):

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if (
            path is None
            or self.current_waypoint_idx <= 0
        ):
            return tx, ty

        px, py = path.waypoints[
            self.current_waypoint_idx - 1
        ]

        dx = tx - px
        dy = ty - py

        dist = np.hypot(dx, dy)

        if dist < 1e-6:
            return tx, ty

        nx = -dy / dist
        ny = dx / dist

        return (
            tx + nx * self.current_lane_offset,
            ty + ny * self.current_lane_offset
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

    def compute_prediction_consistency(self):
        def consistency(hist):
            if len(hist) < 5:
                return 1.0
            latest = hist[-1]
            return (
                sum(
                    x == latest
                    for x in hist
                )
                /
                len(hist)
            )
        self.route_consistency = consistency(
            self.route_history
        )
        self.mode_consistency = consistency(
            self.mode_history
        )
        self.prediction_trust = (
            self.route_consistency +
            self.mode_consistency
        ) * 0.5
        return self.prediction_trust

    def evaluate_scene_risk(self, frame):

        self.minimum_ttc = min(
            (
                self.compute_ttc(o)
                for o in frame["objects"]
                if self.active_track_id is None
                or o["track_id"] == self.active_track_id
            ),
            default=np.inf
        )

        scene = SceneContext(
            timestamp=self.current_timestamp,
            frame_idx=self.frame_idx,
            active_track_id=self.active_track_id,
            frame=frame,
            route_prediction=self.route_info,
            maneuver_prediction=self.mode_info,
            ttc=self.minimum_ttc,
            extra={
                "alert_ood": self.alert_ood,
                "viewer": self,
                "intersection_poly": INTERSECTION_POLY
            }
        )

        self.supervisor_results = (
            self.supervisor_manager.evaluate(
                scene
            )
        )

        self.scenario_results = (
            self.scenario_engine.evaluate(
                scene
            )
        )

        trust_results = {
            r.name: r
            for r in self.supervisor_results
        }

        scenario_results = {
            r.name: r
            for r in self.scenario_results
        }

        ood = trust_results.get(
            "camera_ood"
        )

        track = trust_results.get(
            "track_uncertainty"
        )

        traj = scenario_results.get(
            "trajectory_conflict"
        )

        intersection = scenario_results.get(
            "intersection_occupancy"
        )

        ood_score = getattr(ood, "score", 0.0)
        track_score = getattr(track, "score", 0.0)

        self.current_trust = np.clip(
            1.0
            - 0.35 * ood_score
            - 0.25 * track_score
            - (0.30 if getattr(ood, "flag", False) else 0),
            0.0,
            1.0
        )

        if self.current_trust < 0.25:
            self.runtime_state = "UNSAFE"
        elif self.current_trust < 0.50:
            self.runtime_state = "WARNING"
        elif self.current_trust < 0.80:
            self.runtime_state = "DEGRADED"
        else:
            self.runtime_state = "NORMAL"

        conflict_risk = getattr(traj, "score", 0.0)
        intersection_risk = getattr(
            intersection,
            "score",
            0.0
        )
        self.intersection_risk = (
            intersection_risk
        )
        avoid_side = (
            traj.details.get(
                "avoid_side",
                "LEFT"
            )
            if traj else "LEFT"
        )

        self.update_avoidance_state(
            conflict_risk,
            avoid_side
        )

        trust_issue = (
            self.runtime_state != "NORMAL"
        )

        self.agent_speed_factor = (
            0.5 * self.operator_speed_factor
            if trust_issue
            else self.operator_speed_factor
        )

        if trust_issue:
            self.decision_action = "DEGRADED_MODE"

        elif intersection_risk > 0.5:
            self.decision_action = (
                "WAIT_BEFORE_INTERSECTION"
            )

        elif conflict_risk > 0.75:
            self.decision_action = (
                f"AVOID_{avoid_side}"
            )

        elif conflict_risk > 0.60:
            self.decision_action = "YIELD"

        else:
            self.decision_action = "PROCEED"

        self.warning_level = (
            "EMERGENCY"
            if conflict_risk > 0.90
            else "URGENT"
            if conflict_risk > 0.75
            else "WARNING"
            if conflict_risk > 0.50
            else "NONE"
        )

        self.forklift_alert = (
            conflict_risk > 0.50
        )

        self.scenario_type = (
            "TRUST_DEGRADATION"
            if trust_issue
            else "INTERSECTION_BLOCKAGE"
            if intersection_risk > 0.40
            else "COLLISION_AVOIDANCE"
            if conflict_risk > 0.50
            else "NONE"
        )

        self.scenario_risk = (
            1.0 - self.current_trust
            if trust_issue
            else max(
                conflict_risk,
                intersection_risk
            )
        )

        self.near_miss = (
            self.minimum_ttc < 1.0
            and not self.alert_collision
        )

        self.failed_test = (
            self.alert_collision
            or (
                self.minimum_ttc < 2.0
                and self.decision_action == "PROCEED"
            )
        )

    def compute_ttc(self, obj):

        cx, cy = get_occ_center(obj)

        obj_pos = np.array(
            [cx, cy],
            dtype=float
        )

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

    def draw_robot_path(self):

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if (
            path is None
            or len(path.waypoints) == 0
        ):
            return

        xs = [p[0] for p in path.waypoints]
        ys = [p[1] for p in path.waypoints]

        if self.path_line is None:

            self.path_line, = self.ax.plot(
                xs,
                ys,
                "--",
                color="yellow",
                linewidth=2,
                alpha=0.8,
                zorder=40
            )

            self.path_points = self.ax.scatter(
                xs,
                ys,
                s=60,
                c="red",
                zorder=45
            )

            self.path_target = self.ax.scatter(
                [],
                [],
                s=150,
                c="lime",
                marker="*",
                zorder=50
            )

        else:

            self.path_line.set_data(
                xs,
                ys
            )

            self.path_points.set_offsets(
                np.c_[xs, ys]
            )

        if (
            self.current_waypoint_idx
            < len(path.waypoints)
        ):

            tx, ty = path.waypoints[
                self.current_waypoint_idx
            ]

            self.path_target.set_offsets(
                [[tx, ty]]
            )


    def draw_entry_points(self):

        for artist in self.entry_artists:
            try:
                artist.remove()
            except:
                pass

        self.entry_artists.clear()

        if ENTRY_ELLIPSES is None:
            return

        for entry_id, (center, width, height, angle) in enumerate(ENTRY_ELLIPSES):

            wx, wy = center

            occ = world_to_occupancy(wx, wy)

            if occ is None:
                continue

            px, py = occ

            ellipse = Ellipse(
                xy=(px, py),
                width=120,
                height=80,
                angle=angle,
                fill=False,
                edgecolor="orange",
                linewidth=2.5,
                alpha=0.85
            )

            self.ax.add_patch(ellipse)

            self.entry_artists.append(ellipse)

            text = self.ax.text(
                px,
                py,
                str(entry_id),
                color="orange",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center"
            )

            self.entry_artists.append(text)
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

        idxs = TRACK_HISTORY_IDX.get(tid)

        if idxs is None:
            return

        end_idx = np.searchsorted(
            idxs,
            self.frame_idx,
            side="right"
        )

        history = history[:end_idx]

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

        blend_rate = 4.0

        self.current_lane_offset += np.clip(
            self.target_lane_offset
            - self.current_lane_offset,
            -blend_rate,
            blend_rate
        )

        if (
            self.dragging
            or self.decision_action
            in [
                "WAIT_BEFORE_INTERSECTION",
                "DEGRADED_MODE"
            ]
        ):
            return

        path = DEFAULT_PATHS.get(
            self.current_path_name
        )

        if (
            path is None
            or len(path.waypoints) < 2
        ):
            return

        if (
            self.current_waypoint_idx <= 0
            or
            self.current_waypoint_idx >= len(path.waypoints)
        ):
            self.snap_agent_to_active_path()

        remaining = (
            path.speed
            * self.agent_speed_factor
        )

        while remaining > 0:

            if self.current_waypoint_idx >= len(path.waypoints):
                return

            tx, ty = self.get_offset_target(
                *path.waypoints[
                    self.current_waypoint_idx
                ]
            )

            dx = tx - self.agent_x
            dy = ty - self.agent_y

            dist = math.hypot(dx, dy)

            if dist < 1.0:
                self.current_waypoint_idx += 1
                continue

            step = min(
                remaining,
                dist
            )

            self.agent_x += dx * step / dist
            self.agent_y += dy * step / dist

            remaining -= step

            if step == dist:
                self.current_waypoint_idx += 1
        if abs(self.current_lane_offset) < 0.5:

            path = DEFAULT_PATHS.get(
                self.current_path_name
            )

            if (
                path is not None
                and self.current_waypoint_idx > 0
                and self.current_waypoint_idx < len(path.waypoints)
            ):

                p0 = np.array(
                    path.waypoints[
                        self.current_waypoint_idx - 1
                    ],
                    dtype=float
                )

                p1 = np.array(
                    path.waypoints[
                        self.current_waypoint_idx
                    ],
                    dtype=float
                )

                seg = p1 - p0
                seg_len2 = np.dot(seg, seg)

                if seg_len2 > 1e-6:

                    pos = np.array(
                        [self.agent_x, self.agent_y]
                    )

                    t = np.clip(
                        np.dot(pos - p0, seg) / seg_len2,
                        0.0,
                        1.0
                    )

                    proj = p0 + t * seg

                    self.agent_x = float(proj[0])
                    self.agent_y = float(proj[1])                
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

        #
        # 1. Observe scene
        #
        self.compute_scene_info(
            frame
        )

        #
        # 2. Retrieve predictions
        #
        self.lookup_predictions()

        #
        # 3. Decide
        #
        self.evaluate_scene_risk(
            frame
        )

        #
        # 4. Act
        #
        self.update_agent()

        self.vehicle_dot.set_data(
            [self.agent_x],
            [self.agent_y]
        )

        self.log_evidence()

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

        OOD_MATCH_WINDOW_MS = 1000

        for cam in self.relevant_cameras:

            scores, keys = load_ood(cam)

            if len(keys) == 0:
                continue

            delta = np.abs(
                keys - ts
            )

            idx = delta.argmin()

            #
            # reject stale match
            #

            if delta[idx] > OOD_MATCH_WINDOW_MS:
                continue

            nearest_key = int(
                keys[idx]
            )

            result = scores.get(
                nearest_key
            )

            if result is None:
                continue

            score = float(result[0])

            if score > worst_score:

                worst_score = score

                self.alert_ood = result

        #
        # collision
        #

        objects = frame["objects"]

        if self.active_track_id is not None:
            objects = [
                o for o in objects
                if o["track_id"]
                == self.active_track_id
            ]

        self.alert_collision = False

        for obj in objects:

            cx, cy = get_occ_center(obj)

            d = math.hypot(
                cx - self.agent_x,
                cy - self.agent_y
            )

            if d < 30:
                self.alert_collision = True
                break

    def lookup_predictions(self):
        key = (
            self.current_timestamp,
            self.active_track_id,
        )

        if key == self._last_lookup_key:
            return

        self._last_lookup_key = key        

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
        if pred:

            if (
                pred.get("track_id")
                != self.active_track_id
            ):
                print(
                    "[WARN] TRACK MISMATCH",
                    pred.get("track_id"),
                    self.active_track_id
                )        

        if pred is None:
            return

        self.route_info = pred.get(
            "route"
        )

        self.mode_info = pred
        route = self.route_info

        if route is not None:

            self.route_history.append(
                (
                    int(route.get("from", -1)),
                    int(route.get("to", -1))
                )
            )

        mode_name = pred.get("mode")

        if mode_name is not None:

            self.mode_history.append(
                str(mode_name)
            )
        self.compute_prediction_consistency()

    def draw_bbox(
        self,
        bbox,
        color,
        lw=2,
        alpha=1.0,
        zorder=30
    ):

        artist, = self.ax.plot(
            bbox[:, 0],
            bbox[:, 1],
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=zorder
        )

        self.box_artists.append(
            artist
        )

        return artist

    def draw_velocity_arrow(
        self,
        cx,
        cy,
        angle
    ):

        hx = (
            cx
            + 40 * np.cos(
                np.deg2rad(angle)
            )
        )

        hy = (
            cy
            + 40 * np.sin(
                np.deg2rad(angle)
            )
        )

        arrow, = self.ax.plot(
            [cx, hx],
            [cy, hy],
            color="yellow",
            lw=2,
            zorder=35
        )

        self.box_artists.append(
            arrow
        )
    def add_artist(self, artist):
        self.box_artists.append(artist)
        return artist

    def draw_objects(self, frame):

        visible = set()

        active = self.active_track_id

        for obj in frame["objects"]:

            tid = obj["track_id"]

            visible.add(tid)

            cx, cy = get_occ_center(obj)

            angle = float(obj["angle"])

            width = float(obj["bbox_width"])
            length = float(obj["bbox_length"])

            expand_l, expand_w = uncertainty_bbox(tid)

            inner_box = make_rotated_bbox(
                cx,
                cy,
                width,
                length,
                angle
            )

            outer_box = make_rotated_bbox(
                cx,
                cy,
                width + 2 * expand_w,
                length + 2 * expand_l,
                angle
            )

            outer_artist, inner_artist, arrow_artist, text_artist = (
                self.get_object_artist(tid)
            )

            outer_artist.set_data(
                outer_box[:,0],
                outer_box[:,1]
            )

            inner_artist.set_data(
                inner_box[:,0],
                inner_box[:,1]
            )

            is_active = (
                tid == active
            )

            inner_artist.set_color(
                "lime"
                if is_active
                else "cyan"
            )

            inner_artist.set_linewidth(
                3 if is_active else 2
            )

            vx, vy = obj["velocity_world"]

            if np.hypot(vx,vy) > 0.05:

                hx = (
                    cx +
                    40*np.cos(
                        np.deg2rad(angle)
                    )
                )

                hy = (
                    cy +
                    40*np.sin(
                        np.deg2rad(angle)
                    )
                )

                arrow_artist.set_data(
                    [cx,hx],
                    [cy,hy]
                )

                arrow_artist.set_visible(True)

            else:

                arrow_artist.set_visible(False)

            text_artist.set_position(
                (cx,cy)
            )

            text_artist.set_text(
                str(tid)
            )

            text_artist.set_visible(True)

        #
        # hide objects not visible
        #

        for tid, artists in self.object_artists.items():

            state = (
                tid in visible
            )

            for a in artists:
                a.set_visible(state)

        self.track_ids_current = list(visible)

    def clear_artists(self):
        pass


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
        # ======================================
        # Operational Status
        # ======================================
        self.label_status = QLabel(
            "Collision : NO"
        )

        self.label_runtime = QLabel(
            "Runtime : NORMAL"
        )
        self.label_state = QLabel(
            "System State: NORMAL"
        )

        self.label_scenario = QLabel(
            "Scenario: None"
        )

        self.label_risk = QLabel(
            "Risk Level: LOW"
        )

        # ======================================
        # Mitigation
        # ======================================

        self.label_action = QLabel(
            "Action: PROCEED"
        )

        self.label_warning = QLabel(
            "Warning: NONE"
        )

        # ======================================
        # Trustworthiness
        # ======================================

        self.label_trust = QLabel(
            "Trust Score: 1.00"
        )

        self.label_trust_reason = QLabel(
            "Trust Assessment: OK"
        )

        # ======================================
        # Timing
        # ======================================

        self.label_ttc = QLabel(
            "Predicted TTC: N/A"
        )

        # ======================================
        # Scenario Explanation
        # ======================================

        self.label_explanation = QLabel(
            "No active scenario."
        )

        self.label_explanation.setWordWrap(True)
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

        ctrl.addWidget(self.label_state)

        ctrl.addWidget(self.label_scenario)

        ctrl.addWidget(self.label_risk)

        ctrl.addSpacing(5)

        ctrl.addWidget(self.label_action)

        ctrl.addWidget(self.label_warning)

        ctrl.addSpacing(5)

        ctrl.addWidget(self.label_ttc)

        ctrl.addSpacing(5)

        ctrl.addWidget(self.label_trust)
        ctrl.addWidget(self.label_trust_reason)

        ctrl.addSpacing(10)

        ctrl.addWidget(QLabel("Assessment"))

        ctrl.addWidget(
            self.label_explanation
        )

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
        self._last_visible_tracks = None

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


    def scenario_description(
        self,
        scenario
    ):

        descriptions = {

            "NONE":
            (
                "No hazardous scenario "
                "is currently predicted."
            ),

            "COLLISION_AVOIDANCE":
            (
                "Forklift trajectory is "
                "predicted to intersect "
                "the AMR corridor. "
                "Avoidance manoeuvre "
                "is required."
            ),

            "INTERSECTION_BLOCKAGE":
            (
                "Occupancy prediction "
                "indicates simultaneous "
                "intersection usage. "
                "AMR shall wait before "
                "entering."
            ),

            "TRUST_DEGRADATION":
            (
                "Prediction or perception "
                "confidence below safe "
                "threshold. Reduced "
                "autonomy activated."
            )
        }

        return descriptions.get(
            scenario,
            scenario
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
            new_tid = int(txt)

            if new_tid != self.viewer.active_track_id:

                self.viewer.route_history.clear()
                self.viewer.mode_history.clear()

            self.viewer.active_track_id = new_tid

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

    def set_label(
        self,
        label,
        value
    ):

        label.setText(value)

    def risk_level(
        self,
        risk
    ):

        if risk > 0.80:
            return "CRITICAL"

        if risk > 0.60:
            return "HIGH"

        if risk > 0.30:
            return "MEDIUM"

        return "LOW"

    def trust_description(
        self,
        trust
    ):

        if trust < 0.25:
            return "Unsafe operation"

        if trust < 0.50:
            return "Low confidence"

        if trust < 0.80:
            return "Restricted autonomy"

        return "Nominal operation"

    def update_colored_metric(
        self,
        label,
        value,
        red,
        yellow
    ):

        if value < red:

            color = "#ff4444"

        elif value < yellow:

            color = "#ffb300"

        else:

            color = "#66bb6a"

        label.setStyleSheet(
            f"""
            color:{color};
            font-weight:bold;
            """
        )

    def refresh_info_panel(self):

        v = self.viewer

        route = v.route_info or {}
        mode = v.mode_info or {}

        scenario = getattr(
            v,
            "scenario_type",
            "NONE"
        )

        banner = {

            "NONE":
                ("PROCEED", "#2e7d32"),

            "COLLISION_AVOIDANCE":
                (
                    "AVOIDING FORKLIFT",
                    "#fbc02d"
                ),

            "INTERSECTION_BLOCKAGE":
                (
                    "WAIT AT INTERSECTION",
                    "#fb8c00"
                ),

            "TRUST_DEGRADATION":
                (
                    "DEGRADED MODE",
                    "#d32f2f"
                )
        }

        title, color = banner.get(
            scenario,
            ("UNKNOWN", "#777777")
        )

        self.label_result_banner.setText(
            title
        )

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

        scenario_names = {

            "NONE":
                "No Hazard",

            "COLLISION_AVOIDANCE":
                "Corridor Conflict",

            "INTERSECTION_BLOCKAGE":
                "Intersection Conflict",

            "TRUST_DEGRADATION":
                "Trust Degradation"
        }

        self.set_label(
            self.label_state,
            f"System State: {v.runtime_state}"
        )

        self.set_label(
            self.label_scenario,
            "Scenario: "
            +
            scenario_names.get(
                scenario,
                scenario
            )
        )

        self.set_label(
            self.label_risk,
            f"Risk Level: "
            f"{self.risk_level(v.scenario_risk)}"
        )

        self.set_label(
            self.label_action,
            f"Action: "
            f"{v.decision_action}"
        )

        warning = v.warning_level

        if getattr(
            v,
            "forklift_alert",
            False
        ):
            warning += " | ALERT SENT"

        self.set_label(
            self.label_warning,
            f"Warning: {warning}"
        )

        ttc_text = (
            f"Predicted TTC: "
            f"{v.minimum_ttc:.2f}s"
            if np.isfinite(
                v.minimum_ttc
            )
            else
            "Predicted TTC: N/A"
        )

        self.set_label(
            self.label_ttc,
            ttc_text
        )

        self.set_label(
            self.label_trust,
            f"Trust Score: "
            f"{v.current_trust:.2f}"
        )

        self.set_label(
            self.label_trust_reason,
            "Trust Assessment: "
            +
            self.trust_description(
                v.current_trust
            )
        )

        self.set_label(
            self.label_explanation,
            self.scenario_description(
                scenario
            )
        )

        self.set_label(
            self.label_object,
            f"Forklift ID : "
            f"{v.active_track_id}"
        )

        if route:

            self.set_label(
                self.label_route,
                "Route : "
                +
                (
                    route.get(
                        "display_label"
                    )
                    or
                    f"{route.get('from','?')} → "
                    f"{route.get('to','?')}"
                )
            )

            self.set_label(
                self.label_route_prob,
                f"Route Prob : "
                f"{float(route.get('prob',0)):.2f}"
                if route.get("prob")
                is not None
                else
                "Route Prob : N/A"
            )

            self.set_label(
                self.label_route_unc,
                f"Route Unc : "
                f"{float(route.get('uncertainty',0)):.2f}"
                if route.get("uncertainty")
                is not None
                else
                "Route Unc : N/A"
            )

        else:

            self.set_label(
                self.label_route,
                "Route : N/A"
            )

            self.set_label(
                self.label_route_prob,
                "Route Prob : N/A"
            )

            self.set_label(
                self.label_route_unc,
                "Route Unc : N/A"
            )

        self.set_label(
            self.label_mode,
            f"Maneuver : "
            f"{mode.get('mode','N/A')}"
        )

        conf = mode.get(
            "display_confidence"
        )

        self.set_label(
            self.label_mode_prob,
            (
                f"Maneuver Conf : "
                f"{float(conf):.2f}"
            )
            if conf is not None
            else
            "Maneuver Conf : N/A"
        )

        cams = getattr(
            v,
            "relevant_cameras",
            []
        )

        self.set_label(
            self.label_cam,
            "Camera(s): "
            +
            (
                ",".join(cams)
                if cams
                else "NONE"
            )
        )

        self.set_label(
            self.label_ood,
            (
                f"OOD : "
                f"{float(v.alert_ood[0]):.6f}"
                f" | "
                f"{'ANOMALOUS' if v.alert_ood[1] else 'NORMAL'}"
            )
            if v.alert_ood
            else
            "OOD : N/A"
        )

        self.update_colored_metric(
            self.label_ttc,
            v.minimum_ttc,
            2,
            4
        )

        if v.current_trust < 0.4:

            color = "#ff4444"

        elif v.current_trust < 0.7:

            color = "#ffb300"

        else:

            color = "#66bb6a"

        self.label_trust.setStyleSheet(
            f"""
            color:{color};
            font-weight:bold;
            """
        )

    def refresh_object_combo(self):

        visible = tuple(
            sorted(set(self.viewer.track_ids_current))
        )

        if visible == self._last_visible_tracks:
            return

        self._last_visible_tracks = visible

        selected = self.viewer.active_track_id

        self.object_select.blockSignals(True)

        self.object_select.clear()

        for tid in visible:
            self.object_select.addItem(str(tid))

        if selected in visible:

            idx = self.object_select.findText(
                str(selected)
            )

            if idx >= 0:
                self.object_select.setCurrentIndex(idx)

        elif visible:

            self.viewer.active_track_id = visible[0]
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
        self.viewer.route_history.clear()
        self.viewer.mode_history.clear()        
        self.viewer._last_lookup_key = None
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

        self.viewer.operator_speed_factor = factor


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    sys.exit(app.exec())
