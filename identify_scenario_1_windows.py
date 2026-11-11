#!/usr/bin/env python3
"""Identify diverse recorded forklift interactions near the Scenario 1 intersection."""

import json
import math
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Point, Polygon


REPO_ROOT = Path(__file__).resolve().parent
INTERSECTION_PATH = REPO_ROOT / "scenario_data/intersection_area.json"
TRACKS_PATH = REPO_ROOT / "JSON-data/final.json"
PREDICTIONS_PATH = REPO_ROOT / "JSON-data/predictions_2026_02_27.json"
OUTPUT_JSON = REPO_ROOT / "scenario_data/scenario_1_candidate_windows.json"
OUTPUT_REPORT = REPO_ROOT / "scenario_output/scenario_1_candidate_time_windows.md"
APPROACH_DISTANCE_PX = 160.0
PRE_CONTEXT_FRAMES = 30
POST_CONTEXT_FRAMES = 30
MIN_SEPARATION_FRAMES = 300
TARGET_COUNT = 10
# Exclude one-frame polygon grazes: retained candidates visibly occupy the
# intersection for at least one second of the 10 Hz recording.
MIN_INSIDE_DURATION_S = 1.0


def polygon_for(obj):
    polygon = obj.get("associated_polygon") or []
    return polygon if len(polygon) >= 3 else None


def centre(polygon):
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def side_of_intersection(point, intersection):
    """Coarse human-readable approach/exit side relative to the intersection."""
    centroid = intersection.centroid
    dx = point[0] - centroid.x
    dy = point[1] - centroid.y
    if abs(dx) >= abs(dy):
        return "east" if dx >= 0 else "west"
    return "south" if dy >= 0 else "north"


def duration_bucket(frames):
    if frames < 20:
        return "short (<2 s)"
    if frames < 60:
        return "medium (2–6 s)"
    return "long (≥6 s)"


def load_predictions():
    raw = json.load(PREDICTIONS_PATH.open())
    return {
        int(frame["time_stamp"]): {
            int(prediction["track_id"]): prediction
            for prediction in frame.get("predictions", [])
        }
        for frame in raw
    }


def extract_candidates(frames, predictions, intersection):
    observations = defaultdict(list)
    near_ids_by_frame = []

    for frame_index, frame in enumerate(frames):
        near_ids = set()
        for obj in frame["object_list"]:
            polygon = polygon_for(obj)
            if polygon is None:
                continue
            shape = Polygon(polygon)
            track_id = int(obj["track_id"])
            is_inside = intersection.intersects(shape)
            is_near = is_inside or shape.distance(intersection) <= APPROACH_DISTANCE_PX
            if is_near:
                near_ids.add(track_id)
            observations[track_id].append({
                "frame": frame_index,
                "center": centre(polygon),
                "inside": is_inside,
            })
        near_ids_by_frame.append(near_ids)

    candidates = []
    for track_id, history in observations.items():
        for history_index in range(1, len(history)):
            previous = history[history_index - 1]
            current = history[history_index]
            if current["frame"] != previous["frame"] + 1:
                continue
            if previous["inside"] or not current["inside"]:
                continue

            entry_frame = current["frame"]
            exit_record = None
            last_inside = current
            for later in history[history_index + 1:]:
                if later["frame"] != last_inside["frame"] + 1:
                    break
                if not later["inside"]:
                    exit_record = later
                    break
                last_inside = later
            if exit_record is None:
                continue

            exit_frame = exit_record["frame"]
            inside_frames = exit_frame - entry_frame
            elapsed_s = (
                int(frames[exit_frame]["time_stamp"])
                - int(frames[entry_frame]["time_stamp"])
            ) / 1000.0
            if (inside_frames <= 0 or inside_frames > 300
                    or elapsed_s < MIN_INSIDE_DURATION_S):
                continue

            timestamp = int(frames[entry_frame]["time_stamp"])
            prediction = predictions.get(timestamp, {}).get(track_id, {})
            route = prediction.get("route") or {}
            window_start = max(0, entry_frame - PRE_CONTEXT_FRAMES)
            window_end = min(len(frames) - 1, exit_frame + POST_CONTEXT_FRAMES)
            nearby_ids = set().union(*near_ids_by_frame[window_start:window_end + 1])
            peak_concurrent = max(
                len(ids) for ids in near_ids_by_frame[window_start:window_end + 1]
            )
            candidates.append({
                "focus_track_id": track_id,
                "entry_frame": entry_frame,
                "exit_frame": exit_frame,
                "window_start_frame": window_start,
                "window_end_frame": window_end,
                "entry_timestamp": timestamp,
                "exit_timestamp": int(frames[exit_frame]["time_stamp"]),
                "inside_duration_frames": inside_frames,
                "inside_duration_s": elapsed_s,
                "entry_side": side_of_intersection(previous["center"], intersection),
                "exit_side": side_of_intersection(exit_record["center"], intersection),
                "route": {
                    "from": route.get("from"), "to": route.get("to"),
                    "confidence": route.get("prob"), "uncertainty": route.get("uncertainty"),
                },
                "maneuver": prediction.get("mode", "unknown"),
                "nearby_forklift_ids": sorted(nearby_ids),
                "nearby_forklift_count": len(nearby_ids),
                "peak_concurrent_forklifts": peak_concurrent,
            })
    return candidates


def select_diverse(candidates):
    """Greedily maximize variation in side, duration, route and concurrency."""
    selected = []
    seen = defaultdict(set)
    remaining = list(candidates)
    while remaining and len(selected) < TARGET_COUNT:
        best = None
        best_score = -1
        for candidate in remaining:
            if any(abs(candidate["entry_frame"] - chosen["entry_frame"]) < MIN_SEPARATION_FRAMES
                   for chosen in selected):
                continue
            features = {
                "entry_side": candidate["entry_side"],
                "exit_side": candidate["exit_side"],
                "duration": duration_bucket(candidate["inside_duration_frames"]),
                "route": "{}→{}".format(candidate["route"]["from"], candidate["route"]["to"]),
                "concurrency": min(candidate["peak_concurrent_forklifts"], 3),
            }
            novelty = sum(value not in seen[name] for name, value in features.items())
            score = novelty * 100 + candidate["nearby_forklift_count"] * 5
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        for name, value in {
            "entry_side": best["entry_side"],
            "exit_side": best["exit_side"],
            "duration": duration_bucket(best["inside_duration_frames"]),
            "route": "{}→{}".format(best["route"]["from"], best["route"]["to"]),
            "concurrency": min(best["peak_concurrent_forklifts"], 3),
        }.items():
            seen[name].add(value)
    if len(selected) < TARGET_COUNT:
        raise RuntimeError("Could only select {} diverse windows".format(len(selected)))
    for index, candidate in enumerate(selected, 1):
        candidate["id"] = "scenario_1_window_{:02d}".format(index)
        candidate["diversity_summary"] = (
            "{}→{}; {}; {}; peak {} nearby forklifts".format(
                candidate["entry_side"], candidate["exit_side"],
                duration_bucket(candidate["inside_duration_frames"]),
                "route {}→{}".format(candidate["route"]["from"], candidate["route"]["to"]),
                candidate["peak_concurrent_forklifts"],
            )
        )
    return selected


def write_report(windows):
    lines = [
        "# Scenario 1 — candidate recorded time windows",
        "",
        "These are recorded forklift interactions selected for diversity before AMR paths are assigned. "
        "Each window includes 3 seconds of context before entry and after exit.",
        "",
        "| ID | Source frames | Focus forklift | Movement through intersection | Recorded classifier | Nearby forklifts |",
        "|---|---|---|---|---|---|",
    ]
    for window in windows:
        route = window["route"]
        lines.append(
            f"| {window['id']} | {window['window_start_frame']}–{window['window_end_frame']}; "
            f"entry {window['entry_frame']}, exit {window['exit_frame']} | "
            f"{window['focus_track_id']} | {window['entry_side']} → {window['exit_side']}; "
            f"{window['inside_duration_s']:.1f} s inside | route {route['from']}→{route['to']}, "
            f"{window['maneuver']}; uncertainty {route['uncertainty']} | "
            f"{window['nearby_forklift_count']} distinct, peak {window['peak_concurrent_forklifts']} |"
        )
    lines.extend([
        "",
        "These candidates are not yet AMR scenarios. The next step is to assign only approved reference routes and align each AMR arrival with the recorded interaction to demonstrate entry control, clearance, or fail-safe behaviour.",
    ])
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n")


def main():
    intersection = Polygon(json.load(INTERSECTION_PATH.open())["polygon_image_px"])
    frames = json.load(TRACKS_PATH.open())
    predictions = load_predictions()
    selected = select_diverse(extract_candidates(frames, predictions, intersection))
    payload = {
        "scenario": "Scenario 1 candidate windows",
        "selection_method": "Diverse recorded outside-to-inside forklift transitions near the defined intersection",
        "windows": selected,
    }
    json.dump(payload, OUTPUT_JSON.open("w"), indent=2)
    write_report(selected)
    print("Identified {} candidate recorded windows".format(len(selected)))
    for window in selected:
        print("{}: {}".format(window["id"], window["diversity_summary"]))
    print("Wrote {}".format(OUTPUT_REPORT.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
