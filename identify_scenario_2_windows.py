#!/usr/bin/env python3
"""Select diverse recorded forklift windows for Scenario 2, outside intersections."""

import json
import math
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parent
TRACKS = ROOT / "JSON-data/final.json"
PREDICTIONS = ROOT / "JSON-data/predictions_2026_02_27.json"
INTERSECTION = ROOT / "JSON-data/intersection_area.json"
OUTPUT = ROOT / "JSON-data/scenario_2_candidate_windows.json"
REPORT = ROOT / "report_outputs/scenario_2_candidate_time_windows.md"
WINDOW_HALF = 50  # five seconds at 10 Hz
OUTSIDE_DISTANCE_PX = 180.0
MIN_GLOBAL_SEPARATION = 300


def polygon(obj):
    points = obj.get("associated_polygon") or []
    return points if len(points) >= 3 else None


def center(points):
    return [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]


def speed_band(speed):
    if speed < 5:
        return "stationary/very slow"
    if speed < 20:
        return "slow"
    return "moving"


def main():
    frames = json.load(TRACKS.open())
    prediction_frames = json.load(PREDICTIONS.open())
    predictions = {
        int(frame["time_stamp"]): {int(item["track_id"]): item for item in frame.get("predictions", [])}
        for frame in prediction_frames
    }
    intersection = Polygon(json.load(INTERSECTION.open())["polygon_image_px"])
    history = defaultdict(list)
    frame_centers = []
    for index, frame in enumerate(frames):
        centers = {}
        for obj in frame["object_list"]:
            points = polygon(obj)
            if not points:
                continue
            shape = Polygon(points)
            tid = int(obj["track_id"])
            c = center(points)
            centers[tid] = c
            history[tid].append((index, c, shape.distance(intersection)))
        frame_centers.append(centers)

    candidates = []
    for tid, observations in history.items():
        for pos in range(1, len(observations) - 1):
            frame_index, c, distance = observations[pos]
            previous_index, previous_center, _ = observations[pos - 1]
            if previous_index != frame_index - 1 or distance < OUTSIDE_DISTANCE_PX:
                continue
            if frame_index < WINDOW_HALF or frame_index + WINDOW_HALF >= len(frames):
                continue
            # Keep a sparse set of possible centres from each continuous track.
            if pos % 30:
                continue
            speed = math.hypot(c[0] - previous_center[0], c[1] - previous_center[1]) * 10.0
            timestamp = int(frames[frame_index]["time_stamp"])
            pred = predictions.get(timestamp, {}).get(tid, {})
            route = pred.get("route") or {}
            nearby = sum(
                math.hypot(c[0] - other[0], c[1] - other[1]) <= 220
                for other_id, other in frame_centers[frame_index].items() if other_id != tid
            )
            candidates.append({
                "focus_track_id": tid,
                "window_start_frame": frame_index - WINDOW_HALF,
                "window_center_frame": frame_index,
                "window_end_frame": frame_index + WINDOW_HALF,
                "timestamp": timestamp,
                "distance_to_intersection_px": round(distance, 1),
                "speed_px_s": round(speed, 1),
                "speed_band": speed_band(speed),
                "maneuver": pred.get("mode", "unknown"),
                "route": {"from": route.get("from"), "to": route.get("to"),
                          "uncertainty": route.get("uncertainty")},
                "nearby_forklift_count": nearby,
            })

    selected, seen = [], set()
    while candidates and len(selected) < 10:
        best, best_score = None, -1
        for candidate in candidates:
            if any(abs(candidate["window_center_frame"] - item["window_center_frame"]) < MIN_GLOBAL_SEPARATION for item in selected):
                continue
            features = (
                candidate["maneuver"], candidate["speed_band"],
                "{}→{}".format(candidate["route"]["from"], candidate["route"]["to"]),
                min(candidate["nearby_forklift_count"], 3),
            )
            score = 100 * sum(feature not in seen for feature in features)
            score += min(candidate["nearby_forklift_count"], 3) * 10
            if score > best_score:
                best, best_score = candidate, score
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)
        seen.update((best["maneuver"], best["speed_band"],
                     "{}→{}".format(best["route"]["from"], best["route"]["to"]),
                     min(best["nearby_forklift_count"], 3)))
    if len(selected) < 10:
        raise RuntimeError("Only selected {} Scenario 2 windows".format(len(selected)))
    for index, candidate in enumerate(selected, 1):
        candidate["id"] = "scenario_2_window_{:02d}".format(index)
    json.dump({"scenario": "Scenario 2: operation outside intersections", "selection_method": "Recorded forklift activity at least 180 px from the intersection", "windows": selected}, OUTPUT.open("w"), indent=2)
    lines = ["# Scenario 2 — candidate recorded time windows", "", "Each window is a 10-second recorded forklift segment whose focus track remains at least 180 px from the defined intersection at the centre frame.", "", "| ID | Source frames | Focus forklift behaviour | Prediction | Local traffic |", "|---|---|---|---|---|"]
    for item in selected:
        route = item["route"]
        lines.append(
            f"| {item['id']} | {item['window_start_frame']}–{item['window_end_frame']} | "
            f"track {item['focus_track_id']}; {item['speed_px_s']} px/s ({item['speed_band']}); "
            f"{item['distance_to_intersection_px']} px from intersection | {item['maneuver']}; "
            f"route {route['from']}→{route['to']}; uncertainty {route['uncertainty']} | "
            f"{item['nearby_forklift_count']} nearby |"
        )
    lines.extend(["", "These are recorded scene candidates only. The next step is to assign AMR routes that test TTC monitoring, head-on/crossing avoidance, loading/stationary handling, and fail-safe responses."])
    REPORT.write_text("\n".join(lines) + "\n")
    print("Identified 10 Scenario 2 candidate windows")
    for item in selected:
        print("{}: track {}, {}, {} px/s, {} nearby".format(item["id"], item["focus_track_id"], item["maneuver"], item["speed_px_s"], item["nearby_forklift_count"]))


if __name__ == "__main__":
    main()
