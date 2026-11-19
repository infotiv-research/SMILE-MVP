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

from scenario_config import path


ROOT = Path(__file__).resolve().parent


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return json.load(source)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as target:
        json.dump(payload, target, separators=(",", ":"))


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
    """Adapt replay polygons to the object schema consumed by the MVP viewer."""
    objects = []
    for source in frame.get("objects", []):
        polygon = source.get("polygon", [])
        if len(polygon) < 3:
            continue
        cx = sum(point[0] for point in polygon) / len(polygon)
        cy = sum(point[1] for point in polygon) / len(polygon)
        velocity = source.get("velocity_px_s", [0.0, 0.0])
        objects.append({
            "track_id": int(source["track_id"]),
            "occ_center": (float(cx), float(cy)),
            # The existing MVP TTC implementation reads this field.  Replay
            # input already stores measured BEV velocity in pixels/second.
            "velocity_world": [float(velocity[0]), float(velocity[1])],
            "polygon": polygon,
            "bbox_width": 0.0,
            "bbox_length": 0.0,
            "angle": 0.0,
        })
    return {"timestamp": int(frame["timestamp"]), "objects": objects}


def install_instance(mvp, instance):
    """Install one extracted instance as the existing viewer's data source."""
    mvp.tracks = [adapt_frame(frame) for frame in instance["frames"]]
    mvp.NUM_FRAMES = len(mvp.tracks)
    mvp.prediction_lookup = {
        int(frame["timestamp"]): {
            int(prediction["track_id"]): prediction
            for prediction in frame.get("predictions", [])
        }
        for frame in instance["frames"]
    }
    path_name = "__scenario_replay_path__"
    path = mvp.RobotPath(path_name)
    path.waypoints = [tuple(point) for point in instance["robot_path"]]
    path.speed = float(instance["robot_speed_px_per_frame"])
    mvp.DEFAULT_PATHS[path_name] = path
    return path_name


class ExistingMvpReplay:
    """One stateful, frame-by-frame execution of the existing MVP viewer."""
    def __init__(self, mvp, app, instance):
        self.mvp, self.app, self.instance = mvp, app, instance
        self.start()

    def start(self):
        self.path_name = install_instance(
            self.mvp,
            self.instance
        )

        self.viewer = self.mvp.MVPViewer()

        self.viewer.current_path_name = (
            self.path_name
        )

        focus_track = self.instance.get(
            "focus_track_id"
        )

        self.viewer.active_track_id = (
            int(focus_track)
            if focus_track is not None
            else None
        )

        self.viewer.snap_agent_to_active_path()

        self.cursor = 0
        self.completed = False

        self.positions = []
        self.actions = []

    def reset(self):
        self.viewer.deleteLater()
        self.app.processEvents()
        self.start()

    def advance_to(self, index):
        if index < self.cursor:
            self.reset()
        while self.cursor <= index and self.cursor < len(self.mvp.tracks):
            frame = self.mvp.tracks[self.cursor]
            self.viewer.frame_idx = self.cursor
            self.viewer.current_timestamp = frame["timestamp"]
            self.viewer.compute_scene_info(frame)
            self.viewer.lookup_predictions()
            self.viewer.evaluate_scene_risk(frame)
            if self.completed:
                action = "GOAL_REACHED"
            else:
                self.viewer.update_agent()
                self.completed = self.viewer.current_waypoint_idx >= len(
                    self.mvp.DEFAULT_PATHS[self.path_name].waypoints)
                action = "GOAL_REACHED" if self.completed else self.viewer.decision_action
            self.positions.append(
                [
                    float(self.viewer.agent_x),
                    float(self.viewer.agent_y)
                ]
            )

            snapshot = self.viewer.runtime_snapshot()

            snapshot["action"] = action

            self.actions.append(snapshot)
            self.cursor += 1
        return (self.positions[-1], self.actions[-1])


def run_existing_viewer(mvp, app, instance):
    """
    Run the unmodified MVP decision and motion loop on one instance.
    Produces per-frame diagnostics and aggregate evaluation metrics.
    """

    replay = ExistingMvpReplay(
        mvp,
        app,
        instance
    )

    diagnostics = []

    min_ttc = float("inf")
    min_trust = 1.0
    max_risk = 0.0

    collision = False
    near_miss = False
    failed_test = False

    final_action = "UNKNOWN"
    trigger_action = None
    trigger_ttc = None
    trigger_trust = None
    trigger_frame = None
    for index in range(len(mvp.tracks)):

        replay.advance_to(index)

        snapshot = replay.actions[-1]
        if (
            trigger_action is None
            and snapshot.get("action") in ("YIELD","AVOID_LEFT","WAIT_BEFORE_INTERSECTION")
        ):
            trigger_action = snapshot.get(
                "action"
            )

            trigger_ttc = snapshot.get(
                "ttc"
            )

            trigger_trust = snapshot.get(
                "trust"
            )

            trigger_frame = index
        ttc = snapshot.get("ttc")
        trust = snapshot.get("trust", 1.0)
        risk = snapshot.get("risk", 0.0)

        if ttc is not None:
            min_ttc = min(
                min_ttc,
                float(ttc)
            )

        min_trust = min(
            min_trust,
            float(trust)
        )

        max_risk = max(
            max_risk,
            float(risk)
        )

        collision |= bool(
            snapshot.get("collision")
        )

        near_miss |= bool(
            snapshot.get("near_miss")
        )

        failed_test |= bool(
            snapshot.get("failed_test")
        )

        final_action = snapshot.get(
            "action",
            final_action
        )

        diagnostics.append(
            {
                "frame": index,
                "minimum_ttc_s": ttc,
                "runtime_state": snapshot.get(
                    "runtime_state"
                ),
                "decision_action": snapshot.get(
                    "action"
                ),
                "trust": trust,
                "risk": risk,
            }
        )

    summary = {
        "max_risk": float(max_risk),
        "min_trust": float(min_trust),
        "min_ttc": (
            None
            if min_ttc == float("inf")
            else float(min_ttc)
        ),
        "collision": collision,
        "near_miss": near_miss,
        "failed_test": failed_test,
        "final_action": final_action,
        "trigger_action": trigger_action,
        "trigger_ttc": trigger_ttc,
        "trigger_trust": trigger_trust,
        "trigger_frame": trigger_frame,
        "result": (
            "FAIL"
            if failed_test
            else "PASS"
        ),
    }

    positions = replay.positions
    actions = replay.actions

    replay.viewer.deleteLater()
    app.processEvents()

    return (
        positions,
        actions,
        diagnostics,
        summary,
    )


def artifacts(selected):
    for key in selected:
        compressed = path("scenario_{}_instances".format(key))
        plain = compressed.with_suffix("") if compressed.suffix == ".gz" else compressed
        plain = plain.with_suffix(".json")
        yield plain if plain.exists() else compressed, compressed

def write_summary_csv(
    path,
    rows,
):
    import csv

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

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
        for instance in instances:
            (
                positions,
                actions,
                diagnostics,
                summary,
            ) = run_existing_viewer(
                mvp,
                app,
                instance
            )

            instance["precomputed_behavior"] = {
                "positions": positions,
                "actions": actions,
                "diagnostics": diagnostics,
                "summary": summary,
                "method": (
                    "existing "
                    "tuve_mvp_2026_04_16.MVPViewer "
                    "supervisor and motion loop"
                ),
                "frame_count": len(
                    instance["frames"]
                ),
                "terminal_action": (
                    summary["final_action"]
                ),
            }
            summary_rows.append({
                "instance_id": instance["id"],
                "window_id": instance.get("window_id"),
                "focus_track_id": instance.get(
                    "focus_track_id"
                ),
                "path_label": instance.get(
                    "robot_path_label",
                    ""
                ),
                **summary,
            })            
        destination = args.output_dir / default_destination.name if args.output_dir else default_destination
        write_json(destination, payload)
        print("Wrote {} from the existing MVP supervisor ({} instances)".format(destination, len(instances)))
    if summary_rows:

        csv_path = (
            (args.output_dir or ROOT)
            / "scenario_summary.csv"
        )

        write_summary_csv(
            csv_path,
            summary_rows
        )

        print(f"Wrote {csv_path}")

if __name__ == "__main__":
    main()
