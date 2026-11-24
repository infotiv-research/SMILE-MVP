# ==========================================================
# TUVE MVP – PyQt6 port (compacted, behavior-preserving)
# ==========================================================
import sys, os, json, math, pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Scenario replay is deliberately isolated from the ordinary MVP viewer;
# the default invocation below is unchanged.
if "--scenario" in sys.argv:
    sys.argv.remove("--scenario")
    from scenario_replay_viewer import run as run_scenario_replay_viewer
    sys.exit(run_scenario_replay_viewer())

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from trajectory_player import MLModel
from safety_framework import (
    SceneContext, Supervisor, SupervisorManager, SupervisorResult,
    BBoxUncertaintySupervisor, TrajectoryUncertaintySupervisor,
    RouteUncertaintySupervisor, ManeuverUncertaintySupervisor,
    DecisionEngine,
)

# ==========================================================
# CONFIG
# ==========================================================
CONFIG_FILE = "mvp-config.json"
with open(CONFIG_FILE) as _f:
    CFG = json.load(_f)

DATA_ROOT = Path(CFG["paths"]["dataset_root"])
EXTRA = CFG["extra_paths"]

MASK_FOLDER        = str(DATA_ROOT / EXTRA["mask_folder"])
SAVE_PATH          = str(DATA_ROOT / CFG["paths"]["save_path"])
PRED_PATH          = str(DATA_ROOT / EXTRA["predictions"])
OOD_ROOT           = str(DATA_ROOT / CFG["paths"]["ood_root"])
CAMERA_LOOKUP_PATH = str(DATA_ROOT / CFG["paths"]["camera_lookup_table"])
OCCUPANCY_PATH     = str(DATA_ROOT / EXTRA["occupancy_projection"])
PATH_DB            = str(DATA_ROOT / EXTRA["robot_paths"])
MODEL_PATH         = str(EXTRA["model_path"])

OOD_CACHE = {}

MODEL = MLModel.load(MODEL_PATH)
ENTRY_ELLIPSES = MODEL.entry_ellipses


def load_scenario_polygon(path):
    with open(path) as f:
        return np.asarray(json.load(f)["polygon_image_px"], dtype=float)


SCENARIO_PATHS = CFG["scenario_paths"]
INTERSECTION_AREA_PX    = load_scenario_polygon(Path(SCENARIO_PATHS["intersection_area"]))
ROBOT_OPERATING_AREA_PX = load_scenario_polygon(Path(SCENARIO_PATHS["robot_operating_area"]))

# ==========================================================
# SAFETY POLICY
# ==========================================================
TRUST_TARGET = 0.85
TRUST_UNSAFE = 0.25
FAIL_TTC_SEC = 2.0
TTC_HORIZON_BASE = 4.0
INTERSECTION_LOOKAHEAD_SEC = 15.0


def safety_policy(trust):
    factor = min(1.0, max(0.0, TRUST_TARGET - trust) / max(1e-6, TRUST_TARGET - TRUST_UNSAFE))
    threshold = (1.0 - factor) * TRUST_TARGET
    return {
        "ttc_horizon": TTC_HORIZON_BASE * (1.0 + factor),
        "risk_threshold": threshold,
        "topology_threshold": threshold,
    }


def trust_levels():
    span = TRUST_TARGET - TRUST_UNSAFE
    return {"warning": TRUST_UNSAFE + span / 2.0, "normal": TRUST_TARGET - span / 4.0}

# ==========================================================
# STATIC DATA
# ==========================================================
CACHE_FILE = "tuve_cache.pkl"

mask_lookup = {}
for f in os.listdir(MASK_FOLDER):
    if f.lower().endswith(".jpg"):
        try:
            mask_lookup[int(Path(f).stem)] = f
        except ValueError:
            pass

with open(CAMERA_LOOKUP_PATH, "rb") as f:
    lookup_table = pickle.load(f)

with open(OCCUPANCY_PATH, "rb") as f:
    occ_proj = pickle.load(f)
P = np.asarray(occ_proj["P"], dtype=np.float64)
K = np.asarray(occ_proj["K"], dtype=np.float64)   # kept for API; unused here
R = np.asarray(occ_proj["R"], dtype=np.float64)
t = np.asarray(occ_proj["t"], dtype=np.float64).reshape(3, 1)

# ==========================================================
# HELPERS
# ==========================================================
def world_to_occupancy(wx, wy):
    uvw = (P @ np.array([[wx], [wy], [0.0], [1.0]])).ravel()
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def heading_from_velocity(vx, vy):
    if math.hypot(vx, vy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(vy, vx))


def getcamid(pos):
    return list(lookup_table.get((int(pos[0]), int(pos[1])), {}).keys())


def normalize_cam_id(cam):
    if isinstance(cam, (list, tuple)):
        return str(cam[0]) if cam else None
    return str(cam) if cam else None


def get_occ_center(obj):
    c = obj.get("occ_center")
    if c is None:
        c = world_to_occupancy(*obj["center_world"])
        obj["occ_center"] = c
    return c


def point_in_polygon(point, poly):
    x, y = point
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def make_rotated_bbox(cx, cy, width, length, angle_deg):
    hw, hl = width * 0.5, length * 0.5
    pts = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]])
    a = np.deg2rad(angle_deg)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ Rm.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def world_poly_to_occ(poly):
    if poly is None:
        return None
    return np.asarray([world_to_occupancy(float(x), float(y)) for x, y in np.asarray(poly)])


def load_ood(cam_id):
    cam_id = str(cam_id)
    cached = OOD_CACHE.get(cam_id)
    if cached is not None:
        return cached
    npz = Path(OOD_ROOT) / cam_id / "ood_score.npz"
    if not npz.exists():
        result = ({}, np.empty(0, dtype=np.int64))
    else:
        dtype = np.dtype([("id", np.uint64), ("ood_score", np.float32), ("pred", np.uint8)])
        arr = np.fromfile(npz, dtype=dtype)
        result = ({int(i): (float(s), bool(p)) for i, s, p in arr},
                  arr["id"].astype(np.int64))
    OOD_CACHE[cam_id] = result
    return result

# ==========================================================
# DATA ADAPTORS
# ==========================================================
def build_track_cache(tracks):
    track_unc, track_hist, bbox_stats = {}, {}, {}
    for frame_idx, frame in enumerate(tracks):
        ts = frame["timestamp"]
        for obj in frame["objects"]:
            tid = obj["track_id"]
            cx, cy = world_to_occupancy(*obj["center_world"])
            obj["occ_center"] = (cx, cy)
            track_hist.setdefault(tid, []).append((frame_idx, cx, cy))
            poly = np.asarray(obj["polygon"], dtype=float)
            if len(poly) < 3:
                continue
            center = poly.mean(0)
            track_unc.setdefault(tid, []).append({
                "center": center,
                "area": cv2.contourArea(poly.astype(np.float32)),
                "frame": ts,
            })
            a = np.deg2rad(obj["angle"])
            Rm = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            local = (poly - center) @ Rm.T
            bbox_stats.setdefault(tid, []).append((
                max(0.0, np.ptp(local[:, 0]) - float(obj["bbox_length"])),
                max(0.0, np.ptp(local[:, 1]) - float(obj["bbox_width"])),
            ))
    return track_unc, track_hist, bbox_stats


def load_tracks_auto(path):
    with open(path) as f:
        data = json.load(f)
    if "object_tracks" in data:                      # legacy
        return data["object_tracks"], data.get("uncertainty_boxes"), "legacy"
    if isinstance(data, list) and "object_list" in data[0]:
        return adapt_2026_format(data), None, "2026"
    raise ValueError("Unknown JSON format")


def adapt_2026_format(frames):
    tracks = []
    for frame in frames:
        timestamp = frame.get("time_stamp")
        objects = []
        for obj in frame.get("object_list", []):
            pos = obj.get("position_3d", {})
            vel = obj.get("velocity_3d", {})
            bbox = obj.get("bbox_moving_avg", {})
            heading = obj.get("heading", {})
            angle = heading.get("deg")
            if angle is None:
                angle = heading_from_velocity(float(vel.get("x", 0.0)), float(vel.get("y", 0.0)))
            objects.append({
                "track_id": obj.get("track_id"),
                "timestamp": timestamp,
                "center_world": [float(pos.get("x", 0.0)), float(pos.get("y", 0.0))],
                "velocity_world": [float(vel.get("x", 0.0)), float(vel.get("y", 0.0))],
                "bbox_width": float(bbox.get("width", 0.0)),
                "bbox_length": float(bbox.get("length", 0.0)),
                "polygon": obj.get("associated_polygon", []),
                "angle": float(angle),
            })
        tracks.append({"timestamp": timestamp, "objects": objects})
    return tracks


def estimate_track_uncertainty(frame_idx, track_id):
    recent = [h for h in TRACK_HISTORY.get(track_id, []) if h[0] <= frame_idx][-10:]
    if len(recent) < 3:
        return 0.0
    pts = np.asarray([(h[1], h[2]) for h in recent])
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if len(speed) < 2:
        return 0.0
    mean_speed = float(speed.mean())
    if mean_speed <= 1e-6:
        return 0.0
    return float(np.clip(speed.std() / mean_speed, 0.0, 1.0))

# ==========================================================
# DATA LOAD
# ==========================================================
if Path(CACHE_FILE).exists():
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    tracks = cache["tracks"]
    TRACK_HISTORY, TRACK_UNCERTAINTY, TRACK_BBOX_STATS = cache["history"], cache["unc"], cache["bbox"]
else:
    tracks, _uncertainties, _data_mode = load_tracks_auto(SAVE_PATH)
    TRACK_UNCERTAINTY, TRACK_HISTORY, TRACK_BBOX_STATS = build_track_cache(tracks)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"tracks": tracks, "history": TRACK_HISTORY,
                     "unc": TRACK_UNCERTAINTY, "bbox": TRACK_BBOX_STATS}, f)

TRACK_HISTORY_IDX = {tid: np.asarray([h[0] for h in hist], dtype=np.int32)
                     for tid, hist in TRACK_HISTORY.items()}

first_ts = tracks[0]["timestamp"]
mask_file = mask_lookup.get(first_ts)
if mask_file is None:
    # Headless scenario precompute never renders BEV frames; the normal
    # viewer stays strict.
    if os.environ.get("TUVE_HEADLESS_REPLAY") == "1":
        mask0 = np.zeros((600, 1500), dtype=np.uint8)
    else:
        raise RuntimeError(f"No BEV image for timestamp {first_ts}")
else:
    mask0 = cv2.imread(os.path.join(MASK_FOLDER, mask_file), cv2.IMREAD_GRAYSCALE)
H, W = mask0.shape
NUM_FRAMES = len(tracks)

# ==========================================================
# PRECOMPUTED ROUTE + MANEUVER PREDICTIONS
# ==========================================================
with open(PRED_PATH) as f:
    pred_data = json.load(f)

prediction_lookup = {
    frame.get("time_stamp"): {pred["track_id"]: pred for pred in frame.get("predictions", [])}
    for frame in pred_data
}

# ==========================================================
# BBOX UNCERTAINTY MODEL
# ==========================================================
def uncertainty_bbox(track_id):
    stats = TRACK_BBOX_STATS.get(track_id)
    if not stats:
        return (0.0, 0.0)
    arr = np.asarray(stats)
    return float(np.percentile(arr[:, 0], 95)), float(np.percentile(arr[:, 1], 95))


TRACK_BBOX_MODEL = {}
for tid, samples in TRACK_BBOX_STATS.items():
    arr = np.asarray(samples)
    if len(arr) >= 5:
        TRACK_BBOX_MODEL[tid] = {"mean": arr.mean(0), "cov": np.cov(arr.T)}

# ==========================================================
# LOCAL SUPERVISORS
# ==========================================================
class OODSupervisor(Supervisor):
    name, category = "camera_ood", "input_anomaly"

    def evaluate(self, scene):
        alert = scene.extra.get("alert_ood")
        score = float(alert[0]) if alert is not None else 0.0
        flag = bool(alert[1]) if alert is not None else False
        return SupervisorResult(self.name, self.category, score,
                                max(0.0, 1.0 - score), flag,
                                details={"ood_score": score, "ood_flag": flag})


class LegacyTrackUncertaintySupervisor(Supervisor):
    name, category = "track_uncertainty", "uncertainty"

    def evaluate(self, scene):
        score = 0.0
        if scene.active_track_id is not None:
            score = estimate_track_uncertainty(scene.frame_idx, scene.active_track_id)
        return SupervisorResult(self.name, self.category, score,
                                max(0.0, 1.0 - score), flag=(score > 0.7))

# ==========================================================
# SCENARIOS
# ==========================================================
@dataclass
class ScenarioResult:
    name: str
    score: float
    flag: bool
    details: dict = field(default_factory=dict)


class ScenarioEngine:
    def __init__(self):
        self.scenarios = []

    def register(self, scenario):
        self.scenarios.append(scenario)

    def evaluate(self, scene):
        return [s.evaluate(scene) for s in self.scenarios]


class IntersectionOccupancyScenario:
    name, category = "intersection_occupancy", "scenario"
    OCCUPANCY_TIME = 1.5 * FAIL_TTC_SEC

    @classmethod
    def distance_to_intersection_along_path(cls, viewer, poly_occ):
        path = DEFAULT_PATHS.get(viewer.current_path_name)
        if path is None or viewer.current_waypoint_idx >= len(path.waypoints):
            return np.inf
        route = [(viewer.agent_x, viewer.agent_y), *path.waypoints[viewer.current_waypoint_idx:]]
        travelled = 0.0
        for i in range(len(route) - 1):
            p0, p1 = np.asarray(route[i]), np.asarray(route[i + 1])
            seg = p1 - p0
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue
            steps = max(2, int(seg_len / 25))
            for k in range(steps + 1):
                if point_in_polygon(p0 + seg * k / steps, poly_occ):
                    return travelled
                travelled += seg_len / steps
        return np.inf

    def evaluate(self, scene):
        viewer = scene.extra.get("viewer")
        poly_world = scene.extra.get("INTERSECTION_AREA_PX")
        if viewer is None or poly_world is None:
            return ScenarioResult(self.name, 0.0, False)
        poly_occ = np.asarray(poly_world, dtype=float)
        if len(poly_occ) < 3:
            return ScenarioResult(self.name, 0.0, False)

        center = poly_occ.mean(axis=0)

        robot_distance = self.distance_to_intersection_along_path(viewer, poly_occ)
        if not np.isfinite(robot_distance):
            return ScenarioResult(self.name, 0.0, False)

        path = DEFAULT_PATHS.get(viewer.current_path_name)
        robot_speed = max(0.5, (path.speed if path is not None else 1.0) * viewer.agent_speed_factor)
        robot_eta = robot_distance / robot_speed

        if robot_eta > INTERSECTION_LOOKAHEAD_SEC:
            return ScenarioResult(self.name, 0.0, False, details={"robot_eta": float(robot_eta)})

        best_risk, conflict_track = 0.0, None
        for obj in scene.frame["objects"]:
            cx, cy = get_occ_center(obj)
            vx, vy = obj["velocity_world"]
            speed = np.hypot(vx, vy)
            if speed < 0.2:                     # stationary forklift
                continue
            to_center = center - np.array([cx, cy])
            if np.dot(to_center, [vx, vy]) <= 0:   # not heading toward intersection
                continue
            forklift_eta = np.linalg.norm(to_center) / speed
            risk = np.clip(1.0 - abs(robot_eta - forklift_eta) / self.OCCUPANCY_TIME, 0.0, 1.0)
            if risk > best_risk:
                best_risk, conflict_track = risk, obj["track_id"]

        return ScenarioResult(self.name, float(best_risk), best_risk > 0.5,
                              details={"robot_eta": float(robot_eta),
                                       "occupancy_risk": float(best_risk),
                                       "conflicting_track": conflict_track})


class TrajectoryConflictScenario:
    name, category = "trajectory_conflict", "scenario"
    SAFE_DISTANCE, CRITICAL_DISTANCE, LOOKAHEAD_POINTS = 80, 40, 10

    def evaluate(self, scene):
        viewer = scene.extra.get("viewer")
        if viewer is None:
            return ScenarioResult(self.name, 0.0, False)
        path = DEFAULT_PATHS.get(viewer.current_path_name)
        if path is None or viewer.current_waypoint_idx >= len(path.waypoints):
            return ScenarioResult(self.name, 0.0, False)

        end = min(len(path.waypoints), viewer.current_waypoint_idx + self.LOOKAHEAD_POINTS)
        future = [np.array([viewer.agent_x, viewer.agent_y])]
        future += [np.array(p) for p in path.waypoints[viewer.current_waypoint_idx:end]]

        highest_risk, conflicting, min_dist = 0.0, [], np.inf
        for obj in scene.frame["objects"]:
            cx, cy = get_occ_center(obj)
            pos = np.array([cx, cy])
            if (ROBOT_OPERATING_AREA_PX is not None
                    and not point_in_polygon(pos, ROBOT_OPERATING_AREA_PX)):
                continue
            for p in future:
                d = np.linalg.norm(p - pos)
                if d >= self.SAFE_DISTANCE:
                    continue
                dist_risk = np.clip((self.SAFE_DISTANCE - d)
                                    / (self.SAFE_DISTANCE - self.CRITICAL_DISTANCE), 0.0, 1.0)
                ttc = scene.ttc
                ttc_factor = (np.clip(1.0 - ttc / (5.0 * FAIL_TTC_SEC), 0.0, 1.0)
                              if np.isfinite(ttc) else 1.0)
                risk = dist_risk * ttc_factor
                if risk > 0.25:
                    conflicting.append(obj["track_id"])
                if risk > highest_risk:
                    highest_risk, min_dist = risk, d

        uniq = set(conflicting)
        return ScenarioResult(self.name, float(highest_risk), highest_risk > 0.50,
                              details={"conflicting_tracks": list(uniq),
                                       "conflict_count": len(uniq),
                                       "minimum_distance": float(min_dist),
                                       "avoid_side": "LEFT"})

# ==========================================================
# ROBOT PATH LIBRARY
# ==========================================================
class RobotPath:
    def __init__(self, name):
        self.name, self.speed = name, 2.0
        self.waypoints, self.start_frame = [], 0
        self.end_frame, self.associated_track = None, None


DEFAULT_PATHS = {"Path_1": RobotPath("Path_1")}


def remove_duplicate_waypoints(waypoints, eps=1e-6):
    cleaned = []
    for pt in waypoints:
        if not cleaned or abs(pt[0] - cleaned[-1][0]) > eps or abs(pt[1] - cleaned[-1][1]) > eps:
            cleaned.append(tuple(pt))
    return cleaned


def make_robot_path(name, p, dedup=True):
    path = RobotPath(name)
    path.speed = p.get("speed", 2.0)
    path.start_frame = p.get("start_frame", 0)
    path.end_frame = p.get("end_frame", None)
    path.associated_track = p.get("associated_track", None)
    wps = [tuple(x) for x in p.get("waypoints", [])]
    path.waypoints = remove_duplicate_waypoints(wps) if dedup else wps
    return path


def save_paths():
    data = {n: {"speed": p.speed, "start_frame": p.start_frame, "end_frame": p.end_frame,
                "associated_track": p.associated_track, "waypoints": p.waypoints}
            for n, p in DEFAULT_PATHS.items()}
    with open(PATH_DB, "w") as f:
        json.dump(data, f, indent=2)


def load_paths():
    if os.path.exists(PATH_DB):
        with open(PATH_DB) as f:
            data = json.load(f)
        DEFAULT_PATHS.clear()
        for name, p in data.items():
            DEFAULT_PATHS[name] = make_robot_path(name, p)
    if not DEFAULT_PATHS:
        DEFAULT_PATHS["Path_1"] = RobotPath("Path_1")


load_paths()


# ==========================================================
# VIEWER
# ==========================================================
class MVPViewer(FigureCanvasQTAgg):

    def __init__(self):
        self.fig = Figure(figsize=(8, 8), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.img = self.ax.imshow(mask0, cmap="gray")
        self.vehicle_dot, = self.ax.plot([], [], "bo", markersize=10)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)

        self.entry_artists = []
        self.object_artists = {}
        self.trajectory_artist = None
        self.path_line = self.path_points = self.path_target = None
        self.intersection_patch = self.intersection_label = None
        self.operating_area_patch = None
        self.frame_idx = 0

        self.alert_collision = False
        self.alert_cam = None
        self.alert_ood = None
        self.active_track_id = None
        self.current_lane_offset = 0.0
        self.target_lane_offset = 0.0
        self.route_info = None
        self.mode_info = None
        self.topology_info = {}
        self.display_label = None
        self.track_ids_current = []
        self.current_traj_len = 0
        self.minimum_ttc = 999.0
        self.current_trust = 1.0
        self.runtime_state = "NORMAL"
        self.decision_action = "GO"
        self.near_miss = False
        self.failed_test = False
        self.evidence_log = deque(maxlen=10000)

        self.supervisor_manager = SupervisorManager()
        for s in (OODSupervisor(),
                  LegacyTrackUncertaintySupervisor(),
                  BBoxUncertaintySupervisor(),
                  TrajectoryUncertaintySupervisor(),
                  RouteUncertaintySupervisor(),
                  ManeuverUncertaintySupervisor()):
            self.supervisor_manager.register(s)
        # PlausibilitySupervisor / SafetyMarginSupervisor intentionally disabled.

        self.scenario_engine = ScenarioEngine()
        self.scenario_engine.register(TrajectoryConflictScenario())
        self.scenario_engine.register(IntersectionOccupancyScenario())

        self.supervisor_results = []
        self.decision_engine = DecisionEngine()

        self.route_history = deque(maxlen=10)
        self.mode_history = deque(maxlen=10)
        self.route_consistency = self.mode_consistency = 1.0
        self.prediction_trust = 1.0
        self._last_lookup_key = None

        # ------------------------ AGENT STATE ------------------------
        self.current_path_name = "Path_1" if "Path_1" in DEFAULT_PATHS else next(iter(DEFAULT_PATHS))
        if not DEFAULT_PATHS[self.current_path_name].waypoints:
            DEFAULT_PATHS[self.current_path_name].waypoints = [(100, 240), (700, 240)]
        self.agent_x = self.agent_y = 0.0
        self.agent_start_x = self.agent_start_y = 0.0
        self.agent_speed_factor = 1.0
        self.current_waypoint_idx = 1
        self.snap_agent_to_active_path()

        self.path_edit_mode = False
        self.dragging = False
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_motion)

        self.draw_entry_points()
        self.draw_operating_area()
        self.draw_intersection()

    # ==================================================
    # Ground truth (supervisor-independent)
    # ==================================================
    def evaluate_groundtruth_action(self, frame):
        """Action that SHOULD be taken based purely on physical risk."""
        h = self.evaluate_raw_hazards(frame)
        if h["collision"]:
            return {"expected_action": "EMERGENCY_STOP", "reason": "COLLISION", "hazards": h}
        if h["corridor_conflict_count"] >= 2:
            return {"expected_action": "YIELD", "reason": "CORRIDOR_MULTI", "hazards": h}
        if h["corridor_conflict_count"] == 1:
            return {"expected_action": "AVOID_LEFT", "reason": "CORRIDOR_CONFLICT", "hazards": h}
        if h["intersection_conflict"]:
            return {"expected_action": "YIELD", "reason": "INTERSECTION_BLOCKAGE", "hazards": h}
        return {"expected_action": "PROCEED", "reason": "NONE", "hazards": h}

    def evaluate_raw_hazards(self, frame):
        """Hazard extraction without trust/uncertainty/topology gating."""
        self.minimum_ttc = min((self.compute_ttc(o) for o in frame["objects"]), default=np.inf)
        scene = SceneContext(
            timestamp=self.current_timestamp, frame_idx=self.frame_idx,
            active_track_id=self.active_track_id, frame=frame,
            route_prediction=None, maneuver_prediction=None, ttc=self.minimum_ttc,
            extra={"viewer": self, "INTERSECTION_AREA_PX": INTERSECTION_AREA_PX,
                   "track_history": TRACK_HISTORY, "track_bbox_model": TRACK_BBOX_MODEL},
        )
        results = {r.name: r for r in self.scenario_engine.evaluate(scene)}
        traj, inter = results.get("trajectory_conflict"), results.get("intersection_occupancy")
        conflict_count = traj.details.get("conflict_count", 0) if traj else 0
        return {
            "collision": bool(self.alert_collision),
            "minimum_ttc": float(self.minimum_ttc) if np.isfinite(self.minimum_ttc) else None,
            "corridor_conflict": conflict_count > 0,
            "corridor_conflict_count": conflict_count,
            "corridor_risk": float(traj.score) if traj else 0.0,
            "corridor_tracks": traj.details.get("conflicting_tracks", []) if traj else [],
            "corridor_distance": traj.details.get("minimum_distance") if traj else None,
            "intersection_conflict": (float(inter.score) if inter else 0.0) > 0.0,
            "intersection_risk": float(inter.score) if inter else 0.0,
        }

    # ==================================================
    # Static overlays
    # ==================================================
    def draw_operating_area(self):
        if ROBOT_OPERATING_AREA_PX is None:
            return
        poly = np.asarray(ROBOT_OPERATING_AREA_PX, dtype=float)
        if self.operating_area_patch is None:
            self.operating_area_patch = self.ax.fill(poly[:, 0], poly[:, 1],
                                                     color="limegreen", alpha=0.12, zorder=5)[0]
        else:
            self.operating_area_patch.set_xy(poly)

    def draw_intersection(self):
        if INTERSECTION_AREA_PX is None:
            return
        poly = np.asarray(INTERSECTION_AREA_PX, dtype=float)
        if poly is None:
            return
        risk = getattr(self, "intersection_risk", 0.0)
        color = "red" if risk > 0.50 else "orange" if risk > 0.20 else "gold"
        if self.intersection_patch is None:
            self.intersection_patch = self.ax.fill(poly[:, 0], poly[:, 1],
                                                   color=color, alpha=0.20, zorder=6)[0]
            x, y = poly[:, 0].mean(), poly[:, 1].mean()
            self.intersection_label = self.ax.text(
                x, y, "INTERSECTION", color="darkorange", fontsize=10,
                fontweight="bold", ha="center", va="center",
                bbox=dict(facecolor="black", edgecolor="gold", alpha=0.7), zorder=7)
        else:
            self.intersection_patch.set_xy(poly)
            self.intersection_patch.set_color(color)

    def draw_entry_points(self):
        for a in self.entry_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.entry_artists.clear()
        if ENTRY_ELLIPSES is None:
            return
        for entry_id, (center, _w, _h, angle) in enumerate(ENTRY_ELLIPSES):
            occ = world_to_occupancy(*center)
            if occ is None:
                continue
            px, py = occ
            ellipse = Ellipse(xy=(px, py), width=120, height=80, angle=angle, fill=False,
                              edgecolor="orange", linewidth=2.5, alpha=0.85)
            self.ax.add_patch(ellipse)
            self.entry_artists.append(ellipse)
            self.entry_artists.append(
                self.ax.text(px, py, str(entry_id), color="orange", fontsize=12,
                             fontweight="bold", ha="center", va="center"))

    def get_object_artist(self, tid):
        artist = self.object_artists.get(tid)
        if artist is not None:
            return artist
        outer, = self.ax.plot([], [], color="orange", lw=1.5, alpha=0.6, zorder=25)
        inner, = self.ax.plot([], [], color="cyan", lw=2, zorder=30)
        arrow, = self.ax.plot([], [], color="yellow", lw=2, zorder=35)
        text = self.ax.text(0, 0, "", color="yellow", fontsize=8, ha="center")
        self.object_artists[tid] = (outer, inner, arrow, text)
        return self.object_artists[tid]

    # ==================================================
    # Bookkeeping
    # ==================================================
    def log_evidence(self):
        self.evidence_log.append({
            "frame": int(self.frame_idx), "timestamp": int(self.current_timestamp),
            "ttc": float(self.minimum_ttc), "trust": float(self.current_trust),
            "runtime": self.runtime_state, "decision": self.decision_action,
            "collision": bool(self.alert_collision), "near_miss": bool(self.near_miss),
            "failed": bool(self.failed_test),
        })

    def compute_prediction_consistency(self):
        def consistency(hist):
            if len(hist) < 5:
                return 1.0
            return sum(x == hist[-1] for x in hist) / len(hist)
        self.route_consistency = consistency(self.route_history)
        self.mode_consistency = consistency(self.mode_history)
        self.prediction_trust = (self.route_consistency + self.mode_consistency) * 0.5
        return self.prediction_trust

    # ==================================================
    # Decide
    # ==================================================
    def evaluate_scene_risk(self, frame):
        self.minimum_ttc = min((self.compute_ttc(o) for o in frame["objects"]), default=np.inf)

        mode = self.mode_info or {}
        mode_name = mode.get("mode")
        maneuver_conf = float(mode.get("mode_probs", {}).get(mode_name, 1.0))
        effective_ttc = max(0.0, self.minimum_ttc - FAIL_TTC_SEC * (1.0 - maneuver_conf))

        scene = SceneContext(
            timestamp=self.current_timestamp, frame_idx=self.frame_idx,
            active_track_id=self.active_track_id, frame=frame,
            route_prediction=self.route_info, maneuver_prediction=self.mode_info,
            ttc=effective_ttc,
            extra={"alert_ood": self.alert_ood, "viewer": self,
                   "INTERSECTION_AREA_PX": INTERSECTION_AREA_PX,
                   "track_history": TRACK_HISTORY, "track_bbox_model": TRACK_BBOX_MODEL},
        )
        self.supervisor_results = self.supervisor_manager.evaluate(scene)
        self.scenario_results = self.scenario_engine.evaluate(scene)

        by_name = {r.name: r for r in self.supervisor_results}
        bbox_score = by_name["bbox_uncertainty"].score if "bbox_uncertainty" in by_name else 0.0
        scenarios = {r.name: r for r in self.scenario_results}
        traj = scenarios.get("trajectory_conflict")
        intersection = scenarios.get("intersection_occupancy")
        topology_confidence = float((getattr(self, "topology_info", {}) or {}).get("confidence", 0.0))

        trust = self.decision_engine.compute_trust(self.supervisor_results)
        self.current_trust = trust
        levels = trust_levels()
        if trust < TRUST_UNSAFE:
            self.runtime_state = "UNSAFE"
        elif trust < levels["warning"]:
            self.runtime_state = "WARNING"
        elif trust < levels["normal"]:
            self.runtime_state = "DEGRADED"
        else:
            self.runtime_state = "NORMAL"

        policy = safety_policy(trust)
        ttc_horizon = policy["ttc_horizon"] * (1.0 + 0.5 * bbox_score)
        risk_threshold = policy["risk_threshold"]
        topology_threshold = policy["topology_threshold"]

        conflict_risk = min(1.0, (traj.score if traj else 0.0) * (1.0 + 0.5 * bbox_score))
        if np.isfinite(self.minimum_ttc) and self.minimum_ttc > 2.0 * TTC_HORIZON_BASE:
            conflict_risk *= topology_confidence / max(topology_threshold, 1e-6)

        intersection_risk = intersection.score if intersection else 0.0
        self.intersection_risk = intersection_risk
        conflict_count = traj.details.get("conflict_count", 0) if traj else 0

        if topology_confidence < (1.0 - topology_threshold):
            conflict_count = 0
            conflict_risk *= topology_confidence / max(topology_threshold, 1e-6)
        elif topology_confidence > topology_threshold:
            conflict_risk = min(1.0, conflict_risk * 1.25)

        plausibility = by_name.get("plausibility")
        if plausibility is not None and plausibility.flag:
            conflict_risk = max(conflict_risk, 0.25)

        force_degraded = self.runtime_state == "UNSAFE"
        ttc_intervention = False
        if np.isfinite(effective_ttc) and effective_ttc < ttc_horizon:
            if conflict_count >= 2:
                self.decision_action, ttc_intervention = "YIELD", True
            elif conflict_count == 1:
                self.decision_action, ttc_intervention = "AVOID_LEFT", True

        if not ttc_intervention:
            if force_degraded:
                self.decision_action = "DEGRADED_MODE"
            elif conflict_count >= 2:
                self.decision_action = "YIELD"
            elif conflict_count == 1 and conflict_risk >= risk_threshold:
                self.decision_action = "AVOID_LEFT"
            elif intersection_risk >= risk_threshold:
                self.decision_action = "YIELD"
            else:
                self.decision_action = "PROCEED"

        self.warning_level = ("EMERGENCY" if conflict_risk > 0.90
                              else "WARNING" if conflict_risk > 0.50 else "NONE")
        self.forklift_alert = self.decision_action != "PROCEED"

        if conflict_count > 0 and conflict_risk >= risk_threshold:
            self.scenario_type = "COLLISION_AVOIDANCE"
        elif intersection_risk >= risk_threshold:
            self.scenario_type = "INTERSECTION_BLOCKAGE"
        elif self.runtime_state == "UNSAFE":
            self.scenario_type = "TRUST_DEGRADATION"
        else:
            self.scenario_type = "NONE"

        self.scenario_risk = max(conflict_risk, intersection_risk, 1.0 - trust)
        self.near_miss = self.minimum_ttc < FAIL_TTC_SEC * 0.5 and not self.alert_collision

        intervention = self.decision_action != "PROCEED"
        self.failed_test = self.alert_collision or (effective_ttc < FAIL_TTC_SEC and not intervention)

    def runtime_snapshot(self):
        route = self.route_info or {}
        mode = self.mode_info or {}
        mode_name = mode.get("mode")
        maneuver_conf = mode.get("mode_probs", {}).get(mode_name) if mode_name is not None else None
        return {
            "action": self.decision_action,
            "scenario": getattr(self, "scenario_type", "NONE"),
            "risk": float(getattr(self, "scenario_risk", 0.0)),
            "runtime_state": self.runtime_state,
            "trust": float(self.current_trust),
            "ttc": float(self.minimum_ttc) if np.isfinite(self.minimum_ttc) else None,
            "collision": bool(self.alert_collision),
            "near_miss": bool(self.near_miss),
            "failed_test": bool(self.failed_test),
            "route_prob": route.get("prob"),
            "route_uncertainty": route.get("uncertainty"),
            "maneuver": mode_name,
            "maneuver_confidence": float(maneuver_conf) if maneuver_conf is not None else None,
            "topology_label": getattr(self, "display_label", None),
            "topology_confidence": (getattr(self, "topology_info", {}) or {}).get("confidence"),
        }

    # ==================================================
    # Geometry / TTC
    # ==================================================
    def compute_ttc(self, obj):
        path = DEFAULT_PATHS.get(self.current_path_name)
        if (path is None or self.current_waypoint_idx >= len(path.waypoints)
                or len(tracks) < 2):
            return np.inf
        cx, cy = get_occ_center(obj)

        frame_idx = min(self.frame_idx, len(tracks) - 1)
        if frame_idx == 0:
            dt = (tracks[1]["timestamp"] - tracks[0]["timestamp"]) / 1000.0
        else:
            dt = (tracks[frame_idx]["timestamp"] - tracks[frame_idx - 1]["timestamp"]) / 1000.0
        if dt <= 0:
            return np.inf

        robot_pos = np.array([self.agent_x, self.agent_y], dtype=float)
        tx, ty = path.waypoints[self.current_waypoint_idx]
        heading = np.array([tx - self.agent_x, ty - self.agent_y], dtype=float)
        heading_norm = np.linalg.norm(heading)
        if heading_norm < 1e-6:
            return np.inf
        heading /= heading_norm

        rel_pos = np.array([cx, cy]) - robot_pos
        forward_distance = np.dot(rel_pos, heading)
        if forward_distance <= 0:
            return np.inf

        robot_vel = heading * ((path.speed * self.agent_speed_factor) / dt)
        obj_vel = np.asarray(obj.get("velocity_world", [0.0, 0.0]), dtype=float)
        closing_speed = -np.dot(obj_vel - robot_vel, heading)
        if closing_speed <= 0:
            return np.inf
        return forward_distance / closing_speed

    def get_offset_target(self, tx, ty):
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None or self.current_waypoint_idx <= 0:
            return tx, ty
        px, py = path.waypoints[self.current_waypoint_idx - 1]
        dx, dy = tx - px, ty - py
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return tx, ty
        return (tx + (-dy / dist) * self.current_lane_offset,
                ty + (dx / dist) * self.current_lane_offset)

    def snap_agent_to_active_path(self):
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None or len(path.waypoints) < 2:
            return
        self.agent_x, self.agent_y = map(float, path.waypoints[0])
        self.agent_start_x, self.agent_start_y = self.agent_x, self.agent_y
        self.current_waypoint_idx = 1

    # ==================================================
    # Dynamic drawing
    # ==================================================
    def draw_robot_path(self):
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None or not path.waypoints:
            return
        xs = [p[0] for p in path.waypoints]
        ys = [p[1] for p in path.waypoints]
        if self.path_line is None:
            self.path_line, = self.ax.plot(xs, ys, "--", color="yellow",
                                           linewidth=2, alpha=0.8, zorder=40)
            self.path_points = self.ax.scatter(xs, ys, s=60, c="red", zorder=45)
            self.path_target = self.ax.scatter([], [], s=150, c="lime", marker="*", zorder=50)
        else:
            self.path_line.set_data(xs, ys)
            self.path_points.set_offsets(np.c_[xs, ys])
        if self.current_waypoint_idx < len(path.waypoints):
            tx, ty = path.waypoints[self.current_waypoint_idx]
            self.path_target.set_offsets([[tx, ty]])

    def draw_active_trajectory(self):
        if self.trajectory_artist:
            try:
                self.trajectory_artist.remove()
            except Exception:
                pass
            self.trajectory_artist = None
        self.current_traj_len = 0
        tid = self.active_track_id
        if tid is None:
            return
        idxs = TRACK_HISTORY_IDX.get(tid)
        if idxs is None:
            return
        end = int(np.searchsorted(idxs, self.frame_idx, side="right"))
        history = TRACK_HISTORY.get(tid, [])[:end]
        self.current_traj_len = len(history)
        if len(history) < 2:
            return
        self.trajectory_artist, = self.ax.plot([p[1] for p in history], [p[2] for p in history],
                                               color="yellow", linewidth=3, alpha=0.9, zorder=50)

    def draw_objects(self, frame):
        visible = set()
        for obj in frame["objects"]:
            tid = obj["track_id"]
            visible.add(tid)
            cx, cy = get_occ_center(obj)
            angle = float(obj["angle"])
            width, length = float(obj["bbox_width"]), float(obj["bbox_length"])
            expand_l, expand_w = uncertainty_bbox(tid)
            inner = make_rotated_bbox(cx, cy, width, length, angle)
            outer = make_rotated_bbox(cx, cy, width + 2 * expand_w, length + 2 * expand_l, angle)
            outer_a, inner_a, arrow_a, text_a = self.get_object_artist(tid)
            outer_a.set_data(outer[:, 0], outer[:, 1])
            inner_a.set_data(inner[:, 0], inner[:, 1])
            is_active = tid == self.active_track_id
            inner_a.set_color("lime" if is_active else "cyan")
            inner_a.set_linewidth(3 if is_active else 2)
            vx, vy = obj["velocity_world"]
            if np.hypot(vx, vy) > 0.05:
                hx = cx + 40 * np.cos(np.deg2rad(angle))
                hy = cy + 40 * np.sin(np.deg2rad(angle))
                arrow_a.set_data([cx, hx], [cy, hy])
                arrow_a.set_visible(True)
            else:
                arrow_a.set_visible(False)
            text_a.set_position((cx, cy))
            text_a.set_text(str(tid))
            text_a.set_visible(True)
        for tid, artists in self.object_artists.items():
            for a in artists:
                a.set_visible(tid in visible)
        self.track_ids_current = list(visible)

    def load_bev_image(self, timestamp):
        mask_file = mask_lookup.get(timestamp)
        if mask_file is None:
            return
        self.img.set_data(cv2.imread(os.path.join(MASK_FOLDER, mask_file), cv2.IMREAD_GRAYSCALE))

    # ==================================================
    # Mouse interaction
    # ==================================================
    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self.path_edit_mode:
            path = DEFAULT_PATHS.get(self.current_path_name)
            if path is None:
                return
            if event.button == 1:                                  # add waypoint
                path.waypoints.append((float(event.xdata), float(event.ydata)))
            elif event.button == 3 and path.waypoints:             # delete nearest
                px, py = float(event.xdata), float(event.ydata)
                nearest = min(range(len(path.waypoints)),
                              key=lambda i: (path.waypoints[i][0] - px) ** 2
                                          + (path.waypoints[i][1] - py) ** 2)
                path.waypoints.pop(nearest)
            self.update_frame(self.frame_idx)
            return
        dx, dy = event.xdata - self.agent_x, event.ydata - self.agent_y
        if dx * dx + dy * dy < 400:
            self.dragging = True

    def on_motion(self, event):
        if not self.dragging or event.xdata is None or event.ydata is None:
            return
        self.agent_x, self.agent_y = float(event.xdata), float(event.ydata)
        self.vehicle_dot.set_data([self.agent_x], [self.agent_y])
        self.draw_idle()

    def on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        self.agent_start_x, self.agent_start_y = self.agent_x, self.agent_y
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is not None and len(path.waypoints) >= 2:
            nearest = min(range(len(path.waypoints)),
                          key=lambda i: (path.waypoints[i][0] - self.agent_x) ** 2
                                        + (path.waypoints[i][1] - self.agent_y) ** 2)
            self.current_waypoint_idx = max(1, nearest)
        self.active_track_id = None
        self._last_lookup_key = None

    # ==================================================
    # Agent movement
    # ==================================================
    def update_agent(self):
        self.current_lane_offset += np.clip(self.target_lane_offset - self.current_lane_offset,
                                            -4.0, 4.0)
        if (self.dragging
                or self.decision_action in ("WAIT_BEFORE_INTERSECTION", "DEGRADED_MODE")):
            return
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None or len(path.waypoints) < 2:
            return
        if self.current_waypoint_idx <= 0 or self.current_waypoint_idx >= len(path.waypoints):
            self.snap_agent_to_active_path()

        remaining = path.speed * self.agent_speed_factor
        while remaining > 0:
            if self.current_waypoint_idx >= len(path.waypoints):
                self.current_waypoint_idx = 1
                self.agent_x, self.agent_y = path.waypoints[0]
                self.active_track_id = None
                self._last_lookup_key = None
                best_ttc, replacement = np.inf, None
                for obj in tracks[self.frame_idx]["objects"]:
                    t = self.compute_ttc(obj)
                    if t < best_ttc:
                        best_ttc, replacement = t, obj["track_id"]
                self.active_track_id = replacement
                return
            tx, ty = self.get_offset_target(*path.waypoints[self.current_waypoint_idx])
            dx, dy = tx - self.agent_x, ty - self.agent_y
            dist = math.hypot(dx, dy)
            if dist < 1.0:
                self.current_waypoint_idx += 1
                continue
            step = min(remaining, dist)
            self.agent_x += dx * step / dist
            self.agent_y += dy * step / dist
            remaining -= step
            if step == dist:
                self.current_waypoint_idx += 1

        if abs(self.current_lane_offset) < 0.5:
            if 0 < self.current_waypoint_idx < len(path.waypoints):
                p0 = np.array(path.waypoints[self.current_waypoint_idx - 1], dtype=float)
                p1 = np.array(path.waypoints[self.current_waypoint_idx], dtype=float)
                seg = p1 - p0
                seg_len2 = np.dot(seg, seg)
                if seg_len2 > 1e-6:
                    t = np.clip(np.dot(np.array([self.agent_x, self.agent_y]) - p0, seg) / seg_len2,
                                0.0, 1.0)
                    proj = p0 + t * seg
                    self.agent_x, self.agent_y = float(proj[0]), float(proj[1])

    # ==================================================
    # Scene observation / prediction
    # ==================================================
    def get_relevant_cameras(self):
        path = DEFAULT_PATHS.get(self.current_path_name)
        if path is None or len(path.waypoints) < 2:
            return []
        budget = path.speed * self.agent_speed_factor * 8.0
        cameras = {str(c) for c in getcamid((self.agent_y, self.agent_x))}
        px, py = self.agent_x, self.agent_y
        remaining = budget
        for idx in range(self.current_waypoint_idx, len(path.waypoints)):
            tx, ty = path.waypoints[idx]
            seg_len = math.hypot(tx - px, ty - py)
            if remaining <= 0:
                break
            if seg_len < 1e-6:
                px, py = tx, ty
                continue
            use_len = min(seg_len, remaining)
            samples = max(2, int(use_len / 20))
            ratio = use_len / seg_len
            for k in range(samples + 1):
                a = (k / samples) * ratio
                cameras.update(str(c) for c in
                               getcamid((py + a * (ty - py), px + a * (tx - px))))
            remaining -= use_len
            px, py = tx, ty
        return sorted(cameras)

    def compute_scene_info(self, frame):
        self.alert_collision = False
        self.alert_ood = None
        self.relevant_cameras = self.get_relevant_cameras()
        self.alert_cam = ",".join(self.relevant_cameras) if self.relevant_cameras else None

        ts = int(self.current_timestamp)
        worst = -1.0
        for cam in self.relevant_cameras:
            scores, keys = load_ood(cam)
            if not len(keys):
                continue
            delta = np.abs(keys - ts)
            idx = int(delta.argmin())
            if delta[idx] > 1000:                      # OOD match window (ms)
                continue
            result = scores.get(int(keys[idx]))
            if result is not None and result[0] > worst:
                worst, self.alert_ood = result[0], result

        for obj in frame["objects"]:
            cx, cy = get_occ_center(obj)
            if math.hypot(cx - self.agent_x, cy - self.agent_y) < 30:   # collision px
                self.alert_collision = True
                break

    def lookup_predictions(self):
        key = (self.current_timestamp, self.active_track_id)
        if key == self._last_lookup_key:
            return
        self._last_lookup_key = key
        self.route_info = self.mode_info = None
        if self.active_track_id is None or self.current_timestamp is None:
            return
        pred = prediction_lookup.get(int(self.current_timestamp), {}).get(int(self.active_track_id))
        if pred:
            if pred.get("track_id") != self.active_track_id:
                print("[WARN] TRACK MISMATCH", pred.get("track_id"), self.active_track_id)
        if pred is None:
            return
        self.route_info = pred.get("route")
        self.mode_info = pred
        self.display_label = pred.get("display_label")
        self.topology_info = pred.get("topology", {})

        route = self.route_info
        if route is not None:
            self.route_history.append((int(route.get("from", -1)), int(route.get("to", -1))))
        mode_name = pred.get("mode")
        if mode_name is not None:
            self.mode_history.append(str(mode_name))
        self.compute_prediction_consistency()

    # ==================================================
    # Main frame update
    # ==================================================
    def update_frame(self, idx):
        self.frame_idx = idx
        frame = tracks[idx]
        self.current_timestamp = frame["timestamp"]
        self.load_bev_image(self.current_timestamp)
        self.draw_objects(frame)
        self.compute_scene_info(frame)      # 1. observe
        self.lookup_predictions()           # 2. predict
        self.evaluate_scene_risk(frame)     # 3. decide
        self.update_agent()                 # 4. act
        self.vehicle_dot.set_data([self.agent_x], [self.agent_y])
        self.log_evidence()
        self.draw_active_trajectory()
        self.draw_robot_path()
        self.draw_idle()


# ==========================================================
# MAIN WINDOW + INFO PANEL
# ==========================================================
SCENARIO_META = {
    "NONE":                  ("PROCEED",              "#2e7d32", "No Hazard"),
    "COLLISION_AVOIDANCE":   ("AVOIDING FORKLIFT",    "#fbc02d", "Corridor Conflict"),
    "INTERSECTION_BLOCKAGE": ("WAIT AT INTERSECTION", "#fb8c00", "Intersection Conflict"),
    "TRUST_DEGRADATION":     ("DEGRADED MODE",        "#d32f2f", "Trust Degradation"),
}

SCENARIO_EXPLANATIONS = {
    "NONE": "No hazardous scenario is currently predicted.",
    "COLLISION_AVOIDANCE": ("Forklift trajectory is predicted to intersect the AMR corridor. "
                            "Avoidance manoeuvre is required."),
    "INTERSECTION_BLOCKAGE": ("Occupancy prediction indicates simultaneous intersection usage. "
                              "AMR shall wait before entering."),
    "TRUST_DEGRADATION": ("Prediction or perception confidence below safe threshold. "
                          "Reduced autonomy activated."),
}


class MainWindow(QMainWindow):

    @staticmethod
    def _banner_style(scenario):
        _, color, _ = SCENARIO_META.get(scenario, ("UNKNOWN", "#777777", scenario))
        return (f"background:{color};color:white;font-size:18px;"
                "font-weight:bold;padding:8px;border-radius:4px;")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TUVE MVP – Viewer")
        self.resize(1200, 900)

        self.viewer = MVPViewer()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.next_frame)

        # --- controls
        self.play_btn = QPushButton("▶ Play")
        self.reset_btn = QPushButton("🔄 Reset")
        self.load_btn = QPushButton("📂 Load")
        self.load_scenario2_btn = QPushButton("Load Scenario 2 Test")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, NUM_FRAMES - 1)
        self.new_path_btn = QPushButton("New Path")
        self.save_path_btn = QPushButton("Save Paths")
        self.load_path_btn = QPushButton("Load Paths")
        self.edit_path_btn = QPushButton("Edit Path")
        self.path_select = QComboBox()
        self.path_select.addItems(DEFAULT_PATHS)
        self.window_select = QComboBox()
        self.scenario2_windows = {}
        self.speed_select = QComboBox()
        self.speed_select.addItems(["0.5x", "1x", "2x", "5x", "10x"])
        self.speed_select.setCurrentText("1x")
        self.object_select = QComboBox()
        self.label_frame = QLabel("Frame: 0")

        # --- info labels
        self.label_result_banner = QLabel("PASS")
        self.label_result_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_result_banner.setStyleSheet(self._banner_style("NONE"))
        self.label_state = QLabel("System State: NORMAL")
        self.label_scenario = QLabel("Scenario: None")
        self.label_risk = QLabel("Risk Level: LOW")
        self.label_action = QLabel("Action: PROCEED")
        self.label_warning = QLabel("Warning: NONE")
        self.label_trust = QLabel("Trust Score: 1.00")
        self.label_trust_reason = QLabel("Trust Assessment: OK")
        self.label_ttc = QLabel("Predicted TTC: N/A")
        self.label_explanation = QLabel("No active scenario.")
        self.label_explanation.setWordWrap(True)
        self.label_object = QLabel("Forklift: -")
        self.label_route = QLabel("Route: -")
        self.label_route_prob = QLabel("Route Prob: -")
        self.label_route_unc = QLabel("Route Uncertainty: -")
        self.label_mode = QLabel("Maneuver: -")
        self.label_mode_prob = QLabel("Maneuver Confidence: -")
        self.label_cam = QLabel("Camera: -")
        self.label_ood = QLabel("OOD: -")

        # --- layout
        ctrl = QVBoxLayout()
        for w in (self.load_btn, self.play_btn, self.reset_btn, self.label_frame, self.slider):
            ctrl.addWidget(w)
        ctrl.addWidget(QLabel("AMR Path"))
        for w in (self.path_select, self.new_path_btn, self.save_path_btn, self.load_path_btn,
                  self.edit_path_btn, self.load_scenario2_btn):
            ctrl.addWidget(w)
        ctrl.addWidget(QLabel("Scenario Window"))
        ctrl.addWidget(self.window_select)
        ctrl.addWidget(QLabel("AMR Speed"))
        ctrl.addWidget(self.speed_select)
        ctrl.addWidget(QLabel("Active Forklift"))
        ctrl.addWidget(self.object_select)
        ctrl.addSpacing(10)
        for w in (self.label_result_banner, self.label_state, self.label_scenario, self.label_risk):
            ctrl.addWidget(w)
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
        ctrl.addWidget(self.label_explanation)
        ctrl.addSpacing(10)
        header = QLabel("Prediction")
        header.setStyleSheet("font-weight:bold;")
        ctrl.addWidget(header)
        for w in (self.label_object, self.label_route, self.label_route_prob,
                  self.label_route_unc, self.label_mode, self.label_mode_prob):
            ctrl.addWidget(w)
        ctrl.addSpacing(10)
        diag = QLabel("Diagnostics")
        diag.setStyleSheet("font-weight:bold;")
        ctrl.addWidget(diag)
        ctrl.addWidget(self.label_cam)
        ctrl.addWidget(self.label_ood)
        ctrl.addStretch()

        main = QHBoxLayout()
        main.addWidget(self.viewer, 1)
        side = QWidget()
        side.setLayout(ctrl)
        side.setFixedWidth(250)
        main.addWidget(side)
        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)

        # --- signals
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
        self.load_scenario2_btn.clicked.connect(self.load_scenario2_test_pack)
        self.window_select.currentTextChanged.connect(self.apply_scenario2_window)

    # ------------------- scenario-2 test pack -------------------
    def apply_scenario2_window(self, name):
        w = self.scenario2_windows[name]
        self.viewer.active_track_id = w["focus_track_id"]
        self.viewer._last_lookup_key = None
        frame_idx = w["window_start_frame"]
        self.slider.setValue(frame_idx)
        self.viewer.update_frame(frame_idx)
        self.refresh_info_panel()

    def load_scenario2_test_pack(self):
        with open("./scenario_data/robot_reference_paths_scenario_2.json") as f:
            paths = json.load(f)
        DEFAULT_PATHS.clear()
        for name, p in paths.items():
            DEFAULT_PATHS[name] = make_robot_path(name, p, dedup=False)
        with open("./scenario_data/scenario_2_candidate_windows.json") as f:
            windows = json.load(f)
        self.scenario2_windows = {w["id"]: w for w in windows["windows"]}
        self.path_select.clear()
        self.path_select.addItems(DEFAULT_PATHS)
        self.window_select.clear()
        self.window_select.addItems(self.scenario2_windows)

    # ------------------- path library -------------------
    def save_path_library(self):
        save_paths()

    def load_path_library(self):
        load_paths()
        self.path_select.blockSignals(True)
        self.path_select.clear()
        self.path_select.addItems(DEFAULT_PATHS)
        self.path_select.blockSignals(False)
        if self.path_select.count():
            self.path_select.setCurrentIndex(0)
            self.viewer.current_path_name = self.path_select.currentText()
            self.viewer.snap_agent_to_active_path()
            self.viewer.update_frame(self.viewer.frame_idx)

    def create_new_path(self):
        idx = 1
        while f"Path_{idx}" in DEFAULT_PATHS:
            idx += 1
        name = f"Path_{idx}"
        DEFAULT_PATHS[name] = RobotPath(name)
        self.path_select.addItem(name)
        self.path_select.setCurrentText(name)

    def toggle_path_edit(self):
        self.viewer.path_edit_mode = not self.viewer.path_edit_mode
        path = DEFAULT_PATHS[self.viewer.current_path_name]
        if self.viewer.path_edit_mode:
            self.edit_path_btn.setText("Finish Path")
        else:
            self.edit_path_btn.setText("Edit Path")
            if path.waypoints:
                self.viewer.agent_x, self.viewer.agent_y = path.waypoints[0]
                self.viewer.current_waypoint_idx = 1
            save_paths()
            self.viewer.update_frame(self.viewer.frame_idx)

    def change_path(self, name):
        self.viewer.current_path_name = name
        path = DEFAULT_PATHS.get(name)
        if path is None:
            return
        if path.waypoints:
            self.viewer.snap_agent_to_active_path()
        self.viewer.current_waypoint_idx = 1
        self.viewer.update_frame(self.viewer.frame_idx)
        self.statusBar().showMessage(f"{name}: {len(path.waypoints)} waypoints")

    # ------------------- active forklift -------------------
    def _forward_heading(self):
        path = DEFAULT_PATHS.get(self.viewer.current_path_name)
        if path is None or self.viewer.current_waypoint_idx >= len(path.waypoints):
            return None
        tx, ty = path.waypoints[self.viewer.current_waypoint_idx]
        h = np.array([tx - self.viewer.agent_x, ty - self.viewer.agent_y])
        n = np.linalg.norm(h)
        return h / n if n >= 1e-6 else None

    def choose_active_forklift(self):
        heading = self._forward_heading()
        if heading is None:
            return None
        frame = tracks[self.viewer.frame_idx]
        best_track, best_metric = None, np.inf
        for obj in frame["objects"]:
            cx, cy = get_occ_center(obj)
            rel = np.array([cx - self.viewer.agent_x, cy - self.viewer.agent_y])
            if np.dot(rel, heading) <= 0:
                continue
            ttc = self.viewer.compute_ttc(obj)
            metric = ttc if np.isfinite(ttc) else np.inf
            topo_conf = float(prediction_lookup.get(int(frame["timestamp"]), {})
                                       .get(obj["track_id"], {})
                                       .get("topology", {})
                                       .get("confidence", 0.0))
            if topo_conf > 0.5:
                metric *= 0.75
            if metric < best_metric:
                best_track, best_metric = obj["track_id"], metric
        return best_track

    def active_track_passed(self):
        tid = self.viewer.active_track_id
        if tid is None:
            return True
        frame = tracks[self.viewer.frame_idx]
        obj = next((o for o in frame["objects"] if o["track_id"] == tid), None)
        if obj is None:
            return True
        heading = self._forward_heading()
        if heading is None:
            return False
        cx, cy = get_occ_center(obj)
        rel = np.array([cx - self.viewer.agent_x, cy - self.viewer.agent_y])
        return np.dot(rel, heading) <= 0

    def change_object(self, txt):
        try:
            new_tid = int(txt)
        except ValueError:
            return
        if new_tid != self.viewer.active_track_id:
            self.viewer.route_history.clear()
            self.viewer.mode_history.clear()
        self.viewer.active_track_id = new_tid
        self.viewer.lookup_predictions()
        self.refresh_info_panel()
        self.viewer.draw_idle()

    # ------------------- playback -------------------
    def next_frame(self):
        i = (self.viewer.frame_idx + 1) % NUM_FRAMES
        self.viewer.update_frame(i)
        self.refresh_object_combo()
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.label_frame.setText(f"Frame: {i} | TS: {self.viewer.current_timestamp}")
        self.refresh_info_panel()

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.timer.start()
            self.play_btn.setText("⏸ Pause")

    def reset(self):
        self.timer.stop()
        path = DEFAULT_PATHS[self.viewer.current_path_name]
        if path.waypoints:
            self.viewer.agent_x, self.viewer.agent_y = path.waypoints[0]
            self.viewer.current_waypoint_idx = 1
        self.viewer.frame_idx = 0
        self.viewer.route_history.clear()
        self.viewer.mode_history.clear()
        self.viewer._last_lookup_key = None
        self.viewer.update_frame(0)

    def seek(self, value):
        self.viewer.update_frame(value)
        self.refresh_object_combo()
        self.label_frame.setText(f"Frame: {value} | TS: {self.viewer.current_timestamp}")
        self.refresh_info_panel()

    def change_speed(self, text):
        self.viewer.agent_speed_factor = float(text.replace("x", ""))

    # ------------------- info panel -------------------
    def set_label(self, label, value):
        label.setText(value)

    def risk_level(self, risk):
        if risk > 0.80:
            return "CRITICAL"
        if risk > 0.60:
            return "HIGH"
        if risk > 0.30:
            return "MEDIUM"
        return "LOW"

    def trust_description(self, trust):
        if trust < 0.25:
            return "Unsafe operation"
        if trust < 0.50:
            return "Warning state"
        if trust < 0.70:
            return "Degraded operation"
        return "Nominal operation"

    def update_colored_metric(self, label, value, red, yellow):
        color = "#ff4444" if value < red else "#ffb300" if value < yellow else "#66bb6a"
        label.setStyleSheet(f"color:{color};font-weight:bold;")

    def scenario_description(self, scenario):
        return SCENARIO_EXPLANATIONS.get(scenario, scenario)

    def refresh_info_panel(self):
        v = self.viewer
        scenario = getattr(v, "scenario_type", "NONE")
        title, _color, display_name = SCENARIO_META.get(scenario, ("UNKNOWN", "#777777", scenario))
        self.label_result_banner.setText(title)
        self.label_result_banner.setStyleSheet(self._banner_style(scenario))

        self.set_label(self.label_state, f"System State: {v.runtime_state}")
        self.set_label(self.label_scenario, f"Scenario: {display_name}")
        self.set_label(self.label_risk, f"Risk Level: {self.risk_level(v.scenario_risk)}")
        self.set_label(self.label_action, f"Action: {v.decision_action}")
        warning = v.warning_level + (" | ALERT SENT" if getattr(v, "forklift_alert", False) else "")
        self.set_label(self.label_warning, f"Warning: {warning}")
        ttc = f"{v.minimum_ttc:.2f}s" if np.isfinite(v.minimum_ttc) else "N/A"
        self.set_label(self.label_ttc, f"Predicted TTC: {ttc}")
        self.set_label(self.label_trust, f"Trust Score: {v.current_trust:.2f}")
        self.set_label(self.label_trust_reason,
                       "Trust Assessment: " + self.trust_description(v.current_trust))
        self.set_label(self.label_explanation, self.scenario_description(scenario))
        self.set_label(self.label_object, f"Forklift ID : {v.active_track_id}")

        route = v.route_info or {}
        if route:
            label = getattr(v, "display_label", None)
            if label:
                self.set_label(self.label_route, f"Topology : {label}")
            else:
                self.set_label(self.label_route,
                               f"Route : {route.get('from', '?')} -> {route.get('to', '?')}")
            self.set_label(self.label_route_prob,
                           f"Route Prob : {float(route.get('prob', 0)):.2f}"
                           if route.get("prob") is not None else "Route Prob : N/A")
            self.set_label(self.label_route_unc,
                           f"Route Unc : {float(route.get('uncertainty', 0)):.2f}"
                           if route.get("uncertainty") is not None else "Route Unc : N/A")
        else:
            self.set_label(self.label_route, "Route : N/A")
            self.set_label(self.label_route_prob, "Route Prob : N/A")
            self.set_label(self.label_route_unc, "Route Unc : N/A")

        mode = v.mode_info or {}
        self.set_label(self.label_mode, f"Maneuver : {mode.get('mode', 'N/A')}")
        conf = mode.get("mode_probs", {}).get(mode.get("mode"))
        self.set_label(self.label_mode_prob,
                       f"Maneuver Conf : {float(conf):.2f}" if conf is not None
                       else "Maneuver Conf : N/A")

        cams = getattr(v, "relevant_cameras", [])
        self.set_label(self.label_cam, "Camera(s): " + (",".join(cams) if cams else "NONE"))
        self.set_label(self.label_ood,
                       (f"OOD : {float(v.alert_ood[0]):.6f} | "
                        f"{'ANOMALOUS' if v.alert_ood[1] else 'NORMAL'}")
                       if v.alert_ood else "OOD : N/A")

        self.update_colored_metric(self.label_ttc, v.minimum_ttc, FAIL_TTC_SEC, 2.0 * FAIL_TTC_SEC)
        levels = trust_levels()
        color = ("#ff4444" if v.current_trust < TRUST_UNSAFE
                 else "#ffb300" if v.current_trust < levels["normal"] else "#66bb6a")
        self.label_trust.setStyleSheet(f"color:{color};font-weight:bold;")

    def refresh_object_combo(self):
        visible = tuple(sorted(set(self.viewer.track_ids_current)))
        selected = self.viewer.active_track_id
        if self.active_track_passed():
            replacement = self.choose_active_forklift()
            if replacement is not None:
                selected = replacement
                self.viewer.active_track_id = replacement
                self.viewer._last_lookup_key = None
                self.viewer.lookup_predictions()
        self.object_select.blockSignals(True)
        self.object_select.clear()
        for tid in visible:
            self.object_select.addItem(str(tid))
        if selected in visible:
            idx = self.object_select.findText(str(selected))
            if idx >= 0:
                self.object_select.setCurrentIndex(idx)
        elif visible:
            self.viewer.active_track_id = visible[0]
            self.object_select.setCurrentIndex(0)
        self.object_select.blockSignals(False)


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())