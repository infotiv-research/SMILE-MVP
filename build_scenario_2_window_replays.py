#!/usr/bin/env python3
"""Turn Scenario 2 candidate windows into viewer-compatible recorded replays."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "JSON-data/scenario_2_candidate_windows.json"
TRACKS = ROOT / "JSON-data/final.json"
PREDICTIONS = ROOT / "JSON-data/predictions_2026_02_27.json"
INTERSECTION = ROOT / "JSON-data/intersection_area.json"
OUTPUT = ROOT / "JSON-data/scenario_2_window_replays.json"


def centre(points):
    return [sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)]


def main():
    candidates = json.load(CANDIDATES.open())["windows"]
    tracks = json.load(TRACKS.open())
    prediction_frames = json.load(PREDICTIONS.open())
    predictions = {int(frame["time_stamp"]): frame.get("predictions", []) for frame in prediction_frames}
    instances = []
    for candidate in candidates:
        previous = {}
        replay_frames = []
        for source_index in range(candidate["window_start_frame"], candidate["window_end_frame"] + 1):
            source = tracks[source_index]
            objects = []
            for obj in source["object_list"]:
                polygon = obj.get("associated_polygon") or []
                if len(polygon) < 3:
                    continue
                track_id = int(obj["track_id"])
                current = centre(polygon)
                prior = previous.get(track_id)
                velocity = [0.0, 0.0] if prior is None else [
                    (current[0] - prior[0]) / 0.1, (current[1] - prior[1]) / 0.1,
                ]
                objects.append({"track_id": track_id, "polygon": polygon, "velocity_px_s": velocity})
                previous[track_id] = current
            timestamp = int(source["time_stamp"])
            replay_frames.append({"timestamp": timestamp, "source_frame_index": source_index,
                                  "objects": objects, "predictions": predictions.get(timestamp, [])})
        instances.append({
            "id": candidate["id"], "window_id": candidate["id"],
            "title": "Scenario 2 recorded outside-intersection window (track {})".format(candidate["focus_track_id"]),
            "focus_track_id": candidate["focus_track_id"], "coverage": "recorded_window_only",
            "source_window": {"start_frame": candidate["window_start_frame"], "end_frame": candidate["window_end_frame"]},
            # Placeholder only: Scenario 2 AMR routes are user-authored later.
            "robot_path": [[100.0, 240.0], [700.0, 240.0]], "robot_speed_px_per_frame": 2.0,
            "frames": replay_frames,
        })
    json.dump({"scenario": "Scenario 2: recorded time-window viewer", "intersection_polygon_px": json.load(INTERSECTION.open())["polygon_image_px"], "instances": instances}, OUTPUT.open("w"), indent=2)
    print("Wrote {} with {} viewable windows".format(OUTPUT.relative_to(ROOT), len(instances)))


if __name__ == "__main__":
    main()
