#!/usr/bin/env python3
"""Cache scenario replay behavior by driving the existing MVP supervisor.

This script deliberately contains no safety-decision logic.  It loads each
scenario instance into ``tuve_mvp_2026_04_16.MVPViewer`` and invokes that
module's existing prediction lookup, risk evaluation, and motion methods.
"""

import argparse
from collections import defaultdict
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scenario_config import path
from copy import deepcopy


ROOT = Path(__file__).resolve().parent
FRAME_CACHE = {}
ADAPTED_FRAME_CACHE = {}
PRED_CACHE = {}
VELOCITY_CACHE = {}
BASELINE_VIEWER = None
REPLAY_VIEWER = None
REPLAY_VIEWER_OWNER = None

def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return json.load(source)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as target:
        json.dump(payload, target, separators=(",", ":"))



def build_velocity_cache(instance, horizon_frames=10):
    """
    Precompute hindsight velocities once.

    Current code recomputes future velocity
    every frame and every replay.
    """

    cache = {}

    frames = instance["frames"]

    track_centers = defaultdict(dict)

    for frame_index, frame in enumerate(frames):

        for obj in frame.get("objects", []):

            polygon = obj.get("polygon", [])

            if len(polygon) < 3:
                continue

            center = np.mean(
                np.asarray(polygon, dtype=float),
                axis=0
            )

            track_centers[
                int(obj["track_id"])
            ][frame_index] = center

    for track_id, history in track_centers.items():

        cache[track_id] = {}

        frame_indices = sorted(history)

        for frame_index in frame_indices:

            current_center = history[frame_index]

            best_velocity = [0.0, 0.0]

            end = min(
                len(frames) - 1,
                frame_index + horizon_frames
            )

            for future_index in range(
                end,
                frame_index,
                -1
            ):

                future_center = history.get(
                    future_index
                )

                if future_center is None:
                    continue

                dt = (
                    int(frames[future_index]["timestamp"])
                    -
                    int(frames[frame_index]["timestamp"])
                ) / 1000.0

                if dt <= 0:
                    continue

                velocity = (
                    future_center
                    -
                    current_center
                ) / dt

                best_velocity = [
                    float(velocity[0]),
                    float(velocity[1])
                ]

                break

            cache[track_id][frame_index] = best_velocity

    return cache

def normalise_window_frames(instances):
    """Use the same recorded scene frames for every variant in one window."""
    groups = defaultdict(list)
    for instance in instances:
        groups[instance.get("window_id", instance.get("id"))].append(instance)
    for window_id, variants in groups.items():
        canonical = max(variants, key=lambda item: len(item.get("frames", [])))
        frames = canonical.get("frames", [])
        if not frames:
            raise ValueError("{} has no recorded frames".format(window_id))
        indices = [frame.get("source_frame_index") for frame in frames]
        if any(index is None for index in indices):
            raise ValueError("{} has frame(s) without source indices".format(window_id))
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("{} is not a contiguous recorded window".format(window_id))
        for instance in variants:
            instance["frames"] = list(frames)
            source_window = instance.setdefault("source_window", {})
            source_window["start_frame"] = indices[0]
            source_window["end_frame"] = indices[-1]
            source_window["canonical_window_id"] = window_id


def load_existing_mvp():
    """Import the MVP normally while preventing its explicit scenario GUI mode."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["TUVE_HEADLESS_REPLAY"] = "1"
    original_argv = list(sys.argv)
    sys.argv = [arg for arg in sys.argv if arg != "--scenario"]
    try:
        return importlib.import_module("tuve_mvp_2026_04_16")
    finally:
        sys.argv = original_argv


def adapt_frame(frame):
    ts = int(frame["timestamp"])
    cached = ADAPTED_FRAME_CACHE.get(ts)
    if cached is not None:
        return cached
    objects = []

    for source in frame.get("objects", []):

        polygon = source.get("polygon", [])

        if len(polygon) < 3:
            continue

        poly = np.asarray(
            polygon,
            dtype=float
        )

        center = poly.mean(axis=0)

        #
        # same centroid used by GUI cache building
        #
        cx = float(center[0])
        cy = float(center[1])

        velocity = source.get(
            "velocity_px_s",
            [0.0, 0.0]
        )

        #
        # estimate footprint geometry
        #
        centered = poly - center

        cov = np.cov(
            centered.T
        )

        eigvals, eigvecs = np.linalg.eigh(
            cov
        )

        major_axis = eigvecs[:, np.argmax(eigvals)]

        angle = float(
            np.degrees(
                np.arctan2(
                    major_axis[1],
                    major_axis[0]
                )
            )
        )

        ang = np.deg2rad(angle)

        Rm = np.array(
            [
                [ np.cos(ang),  np.sin(ang)],
                [-np.sin(ang),  np.cos(ang)],
            ]
        )

        local = (
            poly - center
        ) @ Rm.T

        bbox_length = float(
            np.ptp(local[:, 0])
        )

        bbox_width = float(
            np.ptp(local[:, 1])
        )

        objects.append(
            {
                "track_id":
                    int(source["track_id"]),

                "occ_center":
                    (cx, cy),

                "center_world":
                    [cx, cy],

                "velocity_world":
                    [
                        float(velocity[0]),
                        float(velocity[1]),
                    ],

                "bbox_width":
                    bbox_width,

                "bbox_length":
                    bbox_length,

                "polygon":
                    polygon,

                "angle":
                    angle,
            }
        )

    result = {
        "timestamp": int(
            frame["timestamp"]
        ),
        "objects": objects,
    }

    ADAPTED_FRAME_CACHE[
        ts
    ] = result

    return result
def install_instance(mvp, instance):

    canonical = (
        instance
        .setdefault("source_window", {})
        .get("canonical_window_id")
    )

    if canonical not in FRAME_CACHE:

        FRAME_CACHE[canonical] = [
            adapt_frame(frame)
            for frame in instance["frames"]
        ]

    mvp.tracks = FRAME_CACHE[canonical]
    mvp.NUM_FRAMES = len(mvp.tracks)

    if canonical not in PRED_CACHE:

        PRED_CACHE[canonical] = {
            int(frame["timestamp"]): {
                int(pred["track_id"]): pred
                for pred in frame.get(
                    "predictions",
                    []
                )
            }
            for frame in instance["frames"]
        }

    mvp.prediction_lookup = (
        PRED_CACHE[canonical]
    )

    path_name = "__scenario_replay_path__"

    path = mvp.RobotPath(path_name)

    path.waypoints = [
        tuple(point)
        for point in instance["robot_path"]
    ]

    path.speed = float(
        instance["robot_speed_px_per_frame"]
    )

    mvp.DEFAULT_PATHS[path_name] = path

    return path_name


class ExistingMvpReplay:
    """
    Reuses a single MVPViewer across
    all scenario evaluations.

    This removes hundreds of costly:

        MVPViewer()
        FigureCanvas
        SupervisorManager
        ScenarioEngine

    constructions.
    """

    def __init__(
        self,
        mvp,
        app,
        instance,
    ):
        self.mvp = mvp
        self.app = app
        self.instance = instance
        self.start()

    def _reset_viewer_state(
        self,
        viewer,
        focus_track,
    ):

        viewer.frame_idx = 0

        viewer.current_timestamp = None

        viewer.active_track_id = (
            int(focus_track)
            if focus_track is not None
            else None
        )

        viewer.alert_collision = False
        viewer.alert_ood = None

        viewer.route_info = None
        viewer.mode_info = None

        viewer.current_trust = 1.0
        viewer.minimum_ttc = 999.0

        viewer.runtime_state = "NORMAL"
        viewer.decision_action = "PROCEED"

        viewer.near_miss = False
        viewer.failed_test = False

        viewer.route_history.clear()
        viewer.mode_history.clear()

        viewer._last_lookup_key = None

        viewer.snap_agent_to_active_path()

    def start(self):

        global REPLAY_VIEWER
        global REPLAY_VIEWER_OWNER

        self.path_name = install_instance(
            self.mvp,
            self.instance,
        )

        if REPLAY_VIEWER is None:

            REPLAY_VIEWER = (
                self.mvp.MVPViewer()
            )

        self.viewer = REPLAY_VIEWER

        self.viewer.current_path_name = (
            self.path_name
        )

        focus_track = (
            self.instance.get(
                "focus_track_id"
            )
        )

        self._reset_viewer_state(
            self.viewer,
            focus_track,
        )

        REPLAY_VIEWER_OWNER = self

        self.cursor = 0

        self.completed = False

        self.last_position = None
        self.last_snapshot = None

    def reset(self):

        focus_track = (
            self.instance.get(
                "focus_track_id"
            )
        )

        self._reset_viewer_state(
            self.viewer,
            focus_track,
        )

        self.cursor = 0

        self.completed = False
        self.last_position = None
        self.last_snapshot = None

    def advance_to(
        self,
        index,
    ):

        if index < self.cursor:
            self.reset()

        last_position = None
        last_snapshot = None

        while (
            self.cursor <= index
            and
            self.cursor < len(
                self.mvp.tracks
            )
        ):

            frame = (
                self.mvp.tracks[
                    self.cursor
                ]
            )

            self.viewer.frame_idx = (
                self.cursor
            )

            self.viewer.current_timestamp = (
                frame["timestamp"]
            )

            self.viewer.compute_scene_info(
                frame
            )

            self.viewer.lookup_predictions()

            self.viewer.evaluate_scene_risk(
                frame
            )

            if self.completed:

                action = "GOAL_REACHED"

            else:

                self.viewer.update_agent()

                self.completed = (
                    self.viewer.current_waypoint_idx
                    >=
                    len(
                        self.mvp.DEFAULT_PATHS[
                            self.path_name
                        ].waypoints
                    )
                )

                action = (
                    "GOAL_REACHED"
                    if self.completed
                    else self.viewer.decision_action
                )

            last_position = [
                float(
                    self.viewer.agent_x
                ),
                float(
                    self.viewer.agent_y
                ),
            ]

            last_snapshot = (
                self.viewer.runtime_snapshot()
            )

            last_snapshot["action"] = (
                action
            )

            self.cursor += 1

        return (
            last_position,
            last_snapshot,
        )

def _point_segment_distance(point, a, b):
    """Euclidean distance from a point to a line segment."""
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    segment = b - a
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - a))
    projection = float(np.dot(point - a, segment) / length_sq)
    projection = float(np.clip(projection, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + projection * segment)))


def _point_in_polygon(point, polygon):
    """Return True when point is inside a simple polygon."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y))
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def _distance_to_polygon(point, polygon):
    """Distance from point to the observed forklift footprint."""
    if len(polygon) < 3:
        return float("inf")
    polygon = np.asarray(polygon, dtype=float)
    if _point_in_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


def _open_loop_robot_positions(instance):
    """Replay the AMR path at commanded speed without invoking MVP decisions."""
    waypoints = [np.asarray(point, dtype=float) for point in instance["robot_path"]]
    frame_count = len(instance["frames"])
    if len(waypoints) < 2:
        raise ValueError("{} has fewer than two AMR path points".format(instance["id"]))

    speed = float(instance["robot_speed_px_per_frame"])
    position = waypoints[0].copy()
    waypoint_index = 1
    positions = []
    for _ in range(frame_count):
        positions.append([float(position[0]), float(position[1])])
        remaining = speed
        while remaining > 0.0 and waypoint_index < len(waypoints):
            delta = waypoints[waypoint_index] - position
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                waypoint_index += 1
                continue
            step = min(remaining, distance)
            position = position + delta * (step / distance)
            remaining -= step
            if step >= distance - 1e-9:
                waypoint_index += 1
    return positions


def _median_frame_period_seconds(frames):
    timestamps = [int(frame["timestamp"]) for frame in frames]
    periods = [
        (timestamps[i] - timestamps[i - 1]) / 1000.0
        for i in range(1, len(timestamps))
        if timestamps[i] > timestamps[i - 1]
    ]
    return float(np.median(periods)) if periods else 0.1


def _object_by_track(frame, track_id):
    return next(
        (obj for obj in frame.get("objects", []) if int(obj["track_id"]) == int(track_id)),
        None,
    )


def _polygon_center(obj):
    polygon = np.asarray(obj.get("polygon", []), dtype=float)
    if len(polygon) < 3:
        return None
    return polygon.mean(axis=0)


def _future_realized_velocity(frames, frame_index, track_id, horizon_frames=10):
    """Best available realized velocity from now to the end of the test.

    The online MVP sees only the measured velocity at the current frame.  The
    oracle uses subsequent recorded positions, selecting the farthest available
    observation within the look-ahead window.  Units are pixels/second, matching
    the replay adaptor and MVP TTC implementation.
    """
    current = _object_by_track(frames[frame_index], track_id)
    current_center = _polygon_center(current) if current else None
    if current_center is None:
        return [0.0, 0.0]
    current_ts = int(frames[frame_index]["timestamp"])
    end = min(len(frames) - 1, frame_index + int(horizon_frames))
    for future_index in range(end, frame_index, -1):
        future = _object_by_track(frames[future_index], track_id)
        future_center = _polygon_center(future) if future else None
        if future_center is None:
            continue
        dt = (int(frames[future_index]["timestamp"]) - current_ts) / 1000.0
        if dt > 0.0:
            velocity = (future_center - current_center) / dt
            return [float(velocity[0]), float(velocity[1])]
    return [0.0, 0.0]


def _oracle_frame(
    instance,
    frame_index,
    velocity_cache,
):
    cached = adapt_frame(
        instance["frames"][frame_index]
    )

    frame = {
        "timestamp": cached["timestamp"],
        "objects": [
            {
                **obj,
                "velocity_world": list(
                    obj["velocity_world"]
                ),
            }
            for obj in cached["objects"]
        ],
    }

    for obj in frame["objects"]:
        velocity = (
            velocity_cache
            .get(int(obj["track_id"]), {})
            .get(frame_index)
        )

        if velocity is not None:
            obj["velocity_world"] = velocity

    return frame

def _supervisor_value(viewer, name, attribute="score"):
    result = next((r for r in viewer.supervisor_results if r.name == name), None)
    return getattr(result, attribute, None) if result is not None else None


def run_proceed_baseline(
    mvp,
    app,
    instance,
):
    """
    Supervisor-independent oracle.

    Uses:
        evaluate_groundtruth_action()

    Does NOT use:
        evaluate_scene_risk()
        DecisionEngine
        Trust computation
        Supervisors

    Ground truth therefore reflects
    physical hazards only.
    """

    path_name = install_instance(
        mvp,
        instance,
    )

    window_id = (
        instance
        .setdefault("source_window", {})
        .get("canonical_window_id")
    )

    velocity_cache = VELOCITY_CACHE.get(window_id)

    if velocity_cache is None:

        velocity_cache = (
            build_velocity_cache(instance)
        )

        VELOCITY_CACHE[
            window_id
        ] = velocity_cache

    global BASELINE_VIEWER

    if BASELINE_VIEWER is None:

        BASELINE_VIEWER = (
            mvp.MVPViewer()
        )

    viewer = BASELINE_VIEWER
    viewer.route_history.clear()
    viewer.mode_history.clear()
    viewer._last_lookup_key = None
    viewer.snap_agent_to_active_path()
    viewer.current_path_name = (
        path_name
    )

    focus_track = int(
        instance[
            "focus_track_id"
        ]
    )

    viewer.active_track_id = (
        focus_track
    )

    viewer.snap_agent_to_active_path()

    min_ttc = float("inf")

    collision_frame = None
    corridor_frame = None
    intersection_frame = None
    first_required = None

    for index in range(
        len(instance["frames"])
    ):

        frame = _oracle_frame(
            instance,
            index,
            velocity_cache,
        )

        viewer.frame_idx = index

        viewer.current_timestamp = (
            frame["timestamp"]
        )

        viewer.alert_collision = False

        for obj in frame["objects"]:

            cx, cy = obj["occ_center"]

            d = np.hypot(
                cx - viewer.agent_x,
                cy - viewer.agent_y,
            )

            if d < 30:
                viewer.alert_collision = True
                break

        truth = (
            viewer.evaluate_groundtruth_action(
                frame
            )
        )

        hazards = truth[
            "hazards"
        ]

        ttc = hazards.get(
            "minimum_ttc"
        )

        if ttc is not None:
            min_ttc = min(
                min_ttc,
                float(ttc)
            )

        if (
            collision_frame is None
            and hazards["collision"]
        ):
            collision_frame = index
        if (
            corridor_frame is None
            and hazards["corridor_conflict"]
        ):
            corridor_frame = index
        if (
            intersection_frame is None
            and hazards["intersection_conflict"]
        ):
            intersection_frame = index


        expected_action = truth[
            "expected_action"
        ]

        if (
            first_required is None
            and
            expected_action
            != "PROCEED"
        ):
            first_required = {
                "frame":
                    index,

                "action":
                    expected_action,

                "scenario":
                    truth[
                        "reason"
                    ],

                "ttc":
                    hazards.get(
                        "minimum_ttc"
                    ),

                "corridor_conflict_count":
                    hazards.get(
                        "corridor_conflict_count"
                    ),

                "corridor_risk":
                    hazards.get(
                        "corridor_risk"
                    ),

                "intersection_risk":
                    hazards.get(
                        "intersection_risk"
                    ),
            }
            break

        #
        # Continue replay.
        # Ground truth knows the
        # entire scenario.
        #
        viewer.decision_action = (
            "PROCEED"
        )

        viewer.update_agent()
    required = (
        first_required
        or {}
    )

    return {

        "expected_action":
            (
                "INTERVENE"
                if first_required
                else "PROCEED"
            ),

        "ground_truth_basis":
            "raw_hazards_no_supervisors",

        "required_frame":
            required.get("frame"),

        "required_action":
            required.get("action"),

        "required_scenario":
            required.get("scenario"),

        "required_ttc":
            required.get("ttc"),

        "required_risk":
            max(
                required.get(
                    "corridor_risk",
                    0.0,
                ),
                required.get(
                    "intersection_risk",
                    0.0,
                ),
            ),

        "required_corridor_count":
            required.get(
                "corridor_conflict_count"
            ),

        "min_ttc":
            (
                None
                if min_ttc
                == float("inf")
                else min_ttc
            ),

        "collision":collision_frame is not None,
        "near_miss":
            False,

        "collision_frame": collision_frame,
        "corridor_frame": corridor_frame,
        "intersection_frame": intersection_frame,
        "scenarios":
            sorted(
                {
                    "COLLISION"
                    if collision_frame
                    else None,

                    "CORRIDOR_CONFLICT"
                    if corridor_frame
                    else None,

                    "INTERSECTION_BLOCKAGE"
                    if intersection_frame
                    else None,
                }
                - {None}
            ),
    }

def run_existing_viewer(
    mvp,
    app,
    instance,
):
    replay = ExistingMvpReplay(
        mvp,
        app,
        instance,
    )

    trigger = None

    min_ttc = float("inf")
    min_trust = 1.0
    max_risk = 0.0

    frame_count = len(
        mvp.tracks
    )

    for index in range(frame_count):

        _, snapshot = (
            replay.advance_to(index)
        )

        ttc = snapshot.get("ttc")

        trust = float(
            snapshot.get(
                "trust",
                1.0,
            )
        )

        risk = float(
            snapshot.get(
                "risk",
                0.0,
            )
        )

        min_trust = min(
            min_trust,
            trust,
        )

        max_risk = max(
            max_risk,
            risk,
        )

        if ttc is not None:

            min_ttc = min(
                min_ttc,
                float(ttc),
            )

        action = snapshot.get(
            "action"
        )

        if action in {
            "YIELD",
            "AVOID_LEFT",
            "WAIT_BEFORE_INTERSECTION",
            "DEGRADED_MODE",
        }:

            trigger = {
                "frame": index,
                "action": action,
                "state": snapshot.get(
                    "runtime_state"
                ),
                "scenario": snapshot.get(
                    "scenario"
                ),
                "ttc": snapshot.get(
                    "ttc"
                ),
                "trust": snapshot.get(
                    "trust"
                ),
                "route_prob": snapshot.get(
                    "route_prob"
                ),
                "route_uncertainty":
                    snapshot.get(
                        "route_uncertainty"
                    ),
                "maneuver":
                    snapshot.get(
                        "maneuver"
                    ),
                "maneuver_confidence":
                    snapshot.get(
                        "maneuver_confidence"
                    ),
                "collision":
                    snapshot.get(
                        "collision"
                    ),
                "near_miss":
                    snapshot.get(
                        "near_miss"
                    ),
                "risk": risk,
                "topology_label":
                    snapshot.get(
                        "topology_label"
                    ),
                "topology_confidence":
                    snapshot.get(
                        "topology_confidence"
                    ),
            }

            break

    summary = {
        "result": None,
        "outcome": None,
        "final_action":
            (
                trigger["action"]
                if trigger
                else "PROCEED"
            ),
        "max_risk": max_risk,
        "min_trust": min_trust,
        "min_ttc":
            (
                None
                if min_ttc == float("inf")
                else min_ttc
            ),
        "collision":
            bool(
                (trigger or {})
                .get("collision")
            ),
        "near_miss":
            bool(
                (trigger or {})
                .get("near_miss")
            ),
        "failed_test": False,
        "false_positive": False,
        "trigger": trigger,
    }

    #
    # Huge memory saving.
    #
    return [], [], [], summary


def artifacts(selected):
    for key in selected:
        compressed = path("scenario_{}_instances".format(key))
        plain = compressed.with_suffix("") if compressed.suffix == ".gz" else compressed
        plain = plain.with_suffix(".json")
        yield plain if plain.exists() else compressed, compressed

def write_summary_csv(path, rows):

    import csv

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)
def score_against_ground_truth(
    summary,
    baseline,
    fail_ttc,
):
    trigger = (
        summary.get("trigger")
        or {}
    )

    expected = baseline.get(
        "expected_action",
        "PROCEED",
    )

    actual = summary.get(
        "final_action",
        "PROCEED",
    )

    interventions = {
        "YIELD",
        "AVOID_LEFT",
        "WAIT_BEFORE_INTERSECTION",
        "DEGRADED_MODE",
    }

    expected_intervention = (
        expected == "INTERVENE"
    )

    actual_intervention = (
        actual in interventions
    )

    if expected_intervention:

        correct_action = (
            actual_intervention
        )

    else:

        correct_action = (
            actual == "PROCEED"
        )

    required_frame = baseline.get(
        "required_frame"
    )

    trigger_frame = trigger.get(
        "frame"
    )

    if not expected_intervention:

        early_enough = True
        timing_margin_frames = None
        safety_margin = None

    elif not actual_intervention:

        early_enough = False
        timing_margin_frames = None
        safety_margin = None

    elif (
        required_frame is not None
        and trigger_frame is not None
    ):

        early_enough = (
            int(trigger_frame)
            <= int(required_frame)
        )

        timing_margin_frames = (
            int(required_frame)
            - int(trigger_frame)
        )

        safety_margin = None

    else:

        # If we do not know the required frame,
        # but an intervention occurred,
        # assume timing is acceptable.

        early_enough = True
        timing_margin_frames = None
        safety_margin = None

    if (
        expected_intervention
        and not actual_intervention
    ):

        outcome = (
            "MISSED_ACTION"
        )

    elif (
        not expected_intervention
        and actual_intervention
    ):

        outcome = (
            "FALSE_POSITIVE"
        )

    elif not correct_action:

        outcome = (
            "WRONG_ACTION"
        )

    elif not early_enough:

        outcome = (
            "LATE_ACTION"
        )

    else:

        outcome = "PASS"

    summary.update({
        "expected_action":
            expected,

        "actual_action":
            actual,

        "correct_action":
            correct_action,

        "early_enough":
            early_enough,

        "false_positive":
            outcome
            == "FALSE_POSITIVE",

        "outcome":
            outcome,

        "result":
            (
                "PASS"
                if outcome == "PASS"
                else "FAIL"
            ),

        "failed_test":
            outcome != "PASS",

        "planner_score":
            (
                2
                if (
                    correct_action
                    and early_enough
                )
                else 1
                if correct_action
                else 0
            ),

        "timing_margin_frames":
            timing_margin_frames,

        "safety_margin":
            safety_margin,
    })

    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("1", "2"), action="append")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    mvp = load_existing_mvp()
    app = mvp.QApplication.instance() or mvp.QApplication([])
    summary_rows = []
    for source, default_destination in artifacts(args.scenario or ["1", "2"]):
        payload = read_json(source)
        instances = payload.get("instances", [])
        normalise_window_frames(instances)
        expanded = []

        for instance in instances:

            tracks = sorted(
                {
                    int(obj["track_id"])
                    for frame in instance["frames"]
                    for obj in frame.get("objects", [])
                }
            )

            if len(tracks) <= 1:
                expanded.append(instance)
                continue

            for track_id in tracks:

                clone = deepcopy(instance)

                clone["focus_track_id"] = track_id

                clone["id"] = (
                    f"{instance['id']}"
                    f"_track_{track_id}"
                )

                expanded.append(clone)

        instances = expanded       
        for instance in instances:
            baseline = run_proceed_baseline(
                mvp,
                app,
                instance,
            )

            (
                positions,
                actions,
                diagnostics,
                summary,
            ) = run_existing_viewer(
                mvp,
                app,
                instance,
            )

            safety_critical = (
                baseline[
                    "expected_action"
                ]
                == "INTERVENE"
            )
            trigger = summary.get(
                "trigger"
            ) or {}

            expected_action = (
                baseline[
                    "expected_action"
                ]
            )
            summary["baseline"] = baseline
            summary = score_against_ground_truth(
                summary,
                baseline,
                mvp.FAIL_TTC_SEC,
            )
            expected_action = summary[
                "expected_action"
            ]

            instance["precomputed_behavior"] = {
                "positions": positions,
                "actions": actions,
                "diagnostics": diagnostics,
                "summary": summary,
                "method": (
                    "Online MVP supervisor replay scored against the same MVP "
                    "decision policy with full-test hindsight"
                ),
                "frame_count": len(instance["frames"]),
                "terminal_action": (summary["final_action"]),
            }
            summary_rows.append({
                "instance_id": instance["id"],
                "window_id": instance.get("window_id"),
                "focus_track_id": instance.get("focus_track_id"),
                "path_label": instance.get("robot_path_label", ""),
                "result": summary["result"],
                "outcome": summary["outcome"],
                "false_positive": summary["false_positive"],
                "expected_action": summary["expected_action"],
                "actual_action": summary["actual_action"],
                "final_action": summary["final_action"],
                "correct_action": summary["correct_action"],
                "early_enough": summary["early_enough"],
                "planner_score": summary["planner_score"],
                "safety_critical": safety_critical,
                "baseline_required": "YES" if safety_critical else "NO",
                "required_frame": baseline.get("required_frame"),
                "required_action": baseline.get("required_action"),
                "required_scenario": baseline.get("required_scenario"),
                "required_ttc": baseline.get("required_ttc"),
                "required_risk":
                    baseline.get("required_risk"),

                "required_trust":
                    baseline.get("required_trust"),

                "required_bbox_uncertainty":
                    baseline.get(
                        "required_bbox_uncertainty"
                    ),

                "required_route_uncertainty":
                    baseline.get(
                        "required_route_uncertainty"
                    ),

                "required_maneuver_uncertainty":
                    baseline.get(
                        "required_maneuver_uncertainty"
                    ),

                "required_trajectory_uncertainty":
                    baseline.get(
                        "required_trajectory_uncertainty"
                    ),

                "required_track_uncertainty":
                    baseline.get(
                        "required_track_uncertainty"
                    ),

                "required_topology_confidence":
                    baseline.get(
                        "required_topology_confidence"
                    ),                
                "trigger_action": trigger.get("action"),
                "trigger_state": trigger.get("state"),
                "trigger_scenario": trigger.get("scenario"),
                "trigger_frame": trigger.get("frame"),
                "trigger_ttc": trigger.get("ttc"),
                "trigger_trust": trigger.get("trust"),
                "timing_margin_frames": summary.get("timing_margin_frames"),
                "safety_margin": summary.get("safety_margin"),
                "max_risk": summary["max_risk"],
                "min_trust": summary["min_trust"],
                "min_ttc": summary["min_ttc"],
                "collision": summary["collision"],
                "near_miss": summary["near_miss"],
                "failed_test": summary["failed_test"],
                "baseline_min_ttc": baseline.get("min_ttc"),
                "baseline_collision": baseline.get("collision"),
                "ground_truth_basis": baseline.get("ground_truth_basis"),
                "hazard_frame": baseline.get("hazard_frame"),
                "collision_frame": baseline.get("collision_frame"),
                "near_miss_frame": baseline.get("near_miss_frame"),
                "min_clearance_px": baseline.get("min_clearance_px"),
                "min_clearance_frame": baseline.get("min_clearance_frame"),
                "min_clearance_track_id": baseline.get("min_clearance_track_id"),
                "collision_clearance_px": baseline.get("collision_clearance_px"),
                "near_miss_clearance_px": baseline.get("near_miss_clearance_px"),
                "frame_period_sec": baseline.get("frame_period_sec"),
                "route_prob": trigger.get("route_prob"),
                "route_uncertainty": trigger.get("route_uncertainty"),
                "maneuver": trigger.get("maneuver"),
                "maneuver_confidence": trigger.get("maneuver_confidence"),
                "topology_label": trigger.get("topology_label"),
                "topology_confidence": trigger.get("topology_confidence"),
            })
        destination = args.output_dir / default_destination.name if args.output_dir else default_destination
        write_json(destination, payload)
        print("Wrote {} from the existing MVP supervisor ({} instances)".format(destination, len(instances)))
    if summary_rows:
        if args.output_dir:
            csv_path = (args.output_dir/"scenario_summary.csv")
        else:
            csv_path = (default_destination.parent/"scenario_summary.csv")

        write_summary_csv(
            csv_path,
            summary_rows
        )

        print(f"Wrote {csv_path}")

if __name__ == "__main__":
    main()
