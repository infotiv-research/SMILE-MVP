#!/usr/bin/env python3
"""Build Scenario 2 TTC/obstacle replays from recorded outside-intersection windows."""

import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point


ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "JSON-data/scenario_2_candidate_windows.json"
REFERENCES = ROOT / "JSON-data/robot_reference_paths_scenario_2.json"
TRACKS = ROOT / "JSON-data/final.json"
PREDICTIONS = ROOT / "JSON-data/predictions_2026_02_27.json"
INTERSECTION = ROOT / "JSON-data/intersection_area.json"
OUTPUT = ROOT / "JSON-data/scenario_2_instances.json"
REPORT = ROOT / "report_outputs/scenario_2_instance_classification.md"
RELEVANCE_DISTANCE_PX = 60.0


def centre(points):
    return [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]


def sort_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 9999


def segment(waypoints, start, end):
    line = LineString(waypoints)
    result = [list(line.interpolate(start).coords[0])]
    covered = 0.0
    for point_a, point_b in zip(waypoints, waypoints[1:]):
        covered += math.dist(point_a, point_b)
        if start < covered < end:
            result.append([float(point_b[0]), float(point_b[1])])
    finish = list(line.interpolate(end).coords[0])
    if math.dist(result[-1], finish) > 1e-6:
        result.append(finish)
    return result


def focus_samples(candidate, frames):
    samples = []
    for source_index in range(candidate["window_start_frame"], candidate["window_end_frame"] + 1):
        obj = next((item for item in frames[source_index]["object_list"]
                    if int(item["track_id"]) == candidate["focus_track_id"]), None)
        if obj and len(obj.get("associated_polygon", [])) >= 3:
            samples.append((source_index - candidate["window_start_frame"], Point(centre(obj["associated_polygon"]))))
    return samples


def behavior(candidate, minimum_distance, path_index):
    if candidate["maneuver"] == "loading":
        return "loading_obstacle", "Dynamic obstacle avoidance / loading forklift handling"
    if candidate["maneuver"] == "stationary":
        return "stationary_clearance", "Stationary forklift handling: proceed only if no collision risk"
    if path_index < 4 and minimum_distance < 45:
        return "collision_avoidance", "Collision-risk monitoring and dynamic obstacle avoidance"
    if path_index < 7 and minimum_distance < 80:
        return "head_on_or_crossing", "Head-on/crossing interaction: yield, slow, stop, or park safely"
    return "safe_proceed", "Proceed with caution when TTC/collision risk remains acceptable"


def main():
    candidates = json.load(CANDIDATES.open())["windows"]
    references = json.load(REFERENCES.open())
    if len(references) < 10:
        raise ValueError("Scenario 2 needs at least 10 reference paths")
    paths = [(name, references[name]) for name in sorted(references, key=sort_key)]
    frames = json.load(TRACKS.open())
    prediction_frames = json.load(PREDICTIONS.open())
    predictions = {int(item["time_stamp"]): item.get("predictions", []) for item in prediction_frames}
    instances = []
    for window_index, candidate in enumerate(candidates):
        samples = focus_samples(candidate, frames)
        # Only use user routes that spatially overlap the forklift event.
        # A 60 px (~3 m) bound excludes visually unrelated factory regions.
        relevant = []
        for name, definition in paths:
            line = LineString(definition["waypoints"])
            closest = min(
                ((line.distance(point), offset, line.project(point)) for offset, point in samples),
                key=lambda item: item[0],
            )
            if closest[0] <= RELEVANCE_DISTANCE_PX:
                relevant.append((closest, name, definition))
        if not relevant:
            raise ValueError("No Scenario 2 reference path is relevant to {}".format(candidate["id"]))
        relevant.sort(key=lambda item: item[0][0])
        directed = []
        for closest, name, definition in relevant:
            directed.extend([(closest, name, definition, "forward"),
                             (closest, name, definition, "reverse")])
        # When fewer than ten distinct directed paths are relevant, reuse only
        # those geometries with different arrival timing.  This creates ten
        # meaningful interaction variants without admitting remote routes.
        selected = [directed[index % len(directed)] for index in range(10)]
        for path_index, (closest, name, definition, direction) in enumerate(selected):
            original = definition["waypoints"]
            waypoints = original if direction == "forward" else list(reversed(original))
            line = LineString(waypoints)
            # Locate the closest recorded forklift/path pair and time AMR arrival
            # to it, leaving normal motion visible at the beginning of the clip.
            minimum_distance, forklift_offset, _ = closest
            # Recompute projection after reversing the route direction.
            _, _, contact_distance = min(
                ((line.distance(point), offset, line.project(point)) for offset, point in samples),
                key=lambda item: item[0],
            )
            kind, rule = behavior(candidate, minimum_distance, path_index)
            if kind == "safe_proceed":
                target_offset = forklift_offset + 30  # pass after the encounter
            elif kind == "head_on_or_crossing":
                target_offset = forklift_offset + (-8 if path_index % 2 else 8)
            else:
                target_offset = forklift_offset
            target_offset = max(15, min(85, target_offset))
            speed = float(definition.get("speed", 2.0))
            start = max(0.0, contact_distance - speed * target_offset)
            end = min(line.length, contact_distance + speed * 35)
            if end - start < 20:
                start, end = 0.0, line.length
            robot_path = segment(waypoints, start, end)
            previous = {}
            replay_frames = []
            for source_index in range(candidate["window_start_frame"], candidate["window_end_frame"] + 1):
                source = frames[source_index]
                objects = []
                for obj in source["object_list"]:
                    poly = obj.get("associated_polygon") or []
                    if len(poly) < 3:
                        continue
                    tid = int(obj["track_id"])
                    c = centre(poly)
                    prior = previous.get(tid)
                    velocity = [0.0, 0.0] if prior is None else [(c[0] - prior[0]) * 10, (c[1] - prior[1]) * 10]
                    objects.append({"track_id": tid, "polygon": poly, "velocity_px_s": velocity})
                    previous[tid] = c
                timestamp = int(source["time_stamp"])
                replay_frames.append({"timestamp": timestamp, "source_frame_index": source_index,
                                      "objects": objects, "predictions": predictions.get(timestamp, [])})
            instances.append({
                "id": "scenario_2_window_{:02d}_path_{:02d}".format(window_index + 1, path_index + 1),
                "window_id": candidate["id"], "robot_path_id": "robot_path_{:02d}".format(path_index + 1),
                "robot_path_label": "Path {:02d} — {} ({}; timing {:02d})".format(path_index + 1, name, direction, target_offset),
                "title": "{}; track {}; {}".format(candidate["id"], candidate["focus_track_id"], kind),
                "focus_track_id": candidate["focus_track_id"], "coverage": kind,
                "desired_behavior": {"expected_action": kind, "specification_rule": rule,
                                     "minimum_recorded_path_distance_px": round(minimum_distance, 1),
                                     "relevance_threshold_px": RELEVANCE_DISTANCE_PX},
                "robot_reference_path": name, "robot_reference_path_direction": direction,
                "robot_path": robot_path, "robot_speed_px_per_frame": speed,
                "source_window": {"start_frame": candidate["window_start_frame"], "end_frame": candidate["window_end_frame"],
                                  "interaction_frame": candidate["window_start_frame"] + forklift_offset},
                "frames": replay_frames,
            })
    json.dump({
        "scenario": "Scenario 2: operation outside intersections",
        "data_provenance": "Recorded forklift data; simulated AMR paths only.",
        "intersection_polygon_px": json.load(INTERSECTION.open())["polygon_image_px"],
        "instances": instances,
    }, OUTPUT.open("w"), indent=2)
    lines = ["# Scenario 2 — robot-path behavior coverage", "", "Each recorded outside-intersection window has ten AMR route variants. Every selected route comes within 60 px of the recorded focus forklift trajectory; when fewer than ten directed paths are available, only relevant geometry is reused with different arrival timing.", "", "| Window | Coverage |", "|---|---|"]
    for candidate in candidates:
        group = [item for item in instances if item["window_id"] == candidate["id"]]
        labels = ", ".join(item["coverage"] for item in group)
        lines.append("| {} | {} |".format(candidate["id"], labels))
    lines.extend(["", "These variants exercise the existing online TTC/trajectory safety layer. Loading/stationary entries are recorded maneuver contexts; a dedicated Scenario 2 maneuver policy can be added before measuring rule-specific pass/fail results."])
    REPORT.write_text("\n".join(lines) + "\n")
    print("Wrote {} with {} windows × {} paths".format(OUTPUT.relative_to(ROOT), len(candidates), len(paths)))


if __name__ == "__main__":
    main()
