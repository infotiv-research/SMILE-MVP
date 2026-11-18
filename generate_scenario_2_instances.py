#!/usr/bin/env python3
"""Build Scenario 2 TTC/obstacle replays from recorded outside-intersection windows."""

import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon

from scenario_config import path


ROOT = Path(__file__).resolve().parent
CANDIDATES = path("scenario_2_candidate_windows")
REFERENCES = path("scenario_2_reference_paths")
TRACKS = path("recorded_tracks")
PREDICTIONS = path("predictions")
INTERSECTION = path("intersection_area")
OPERATING_AREA = path("robot_operating_area")
OUTPUT = path("scenario_2_instances_json")
REPORT = path("scenario_2_classification_report")
RELEVANCE_DISTANCE_PX = 60.0
WINDOW_6_REFERENCE_PATHS = {"Path_7", "Path_8", "Path_9", "Path_10"}
LONG_LEFT_CONTEXT_WINDOWS = {"scenario_2_window_07", "scenario_2_window_09"}
MIN_START_SEPARATION_PX = 30.0
MIN_ROBOT_PATH_LENGTH_PX = 200.0  # approximately 10 m at 20 px/m
MAX_FOCUS_ENCOUNTER_DISTANCE_PX = 80.0


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


def select_directed_paths(candidate, relevant, samples, frame_count):
    """Select ten distinct directed route/timing variants."""
    directed = []
    for closest, name, definition in relevant:
        directed.extend([(closest, name, definition, "forward"),
                         (closest, name, definition, "reverse")])
    feasible = []
    for closest, name, definition, direction in directed:
        waypoints = (definition["waypoints"] if direction == "forward"
                     else list(reversed(definition["waypoints"])))
        line = LineString(waypoints)
        contact = min(
            ((line.distance(point), line.project(point))
             for _offset, point in samples),
            key=lambda item: item[0],
        )[1]
        # A 15 m route must both contain the contact and let the AMR reach it
        # during this recorded window at the approved nominal speed.
        speed = float(definition.get("speed", 2.0))
        earliest_start = max(0.0, contact - speed * (frame_count - 1))
        latest_start = min(contact, line.length - MIN_ROBOT_PATH_LENGTH_PX)
        if earliest_start <= latest_start:
            feasible.append((closest, name, definition, direction))
    directed = feasible
    if not directed:
        raise ValueError("No 15 m route can reach the focus during {}".format(
            candidate["id"]
        ))
    if candidate["id"] == "scenario_2_window_06":
        directed = [item for item in directed if item[1] in WINDOW_6_REFERENCE_PATHS]
        if not directed:
            raise ValueError("Window 6 has no valid left-side operating-area routes")
        # Reuse the outer Path 7 lane for the final two variants. It preserves
        # the left-side encounter, unlike repeating the centre lane.
        path_7 = [item for item in directed if item[1] == "Path_7"]
        forward_path_7 = [item for item in path_7 if item[3] == "forward"]
        return ([item + (0,) for item in directed] +
                [forward_path_7[0] + (1,), forward_path_7[0] + (2,)])
    # Use each relevant geometry/direction once first. When fewer than ten are
    # available, reuse only with an explicit distinct timing/segment variant.
    # This prevents visually coincident AMRs at replay start.
    return [item + (index // len(directed),) for index, item in
            enumerate(directed[index % len(directed)] for index in range(10))]


def robot_segment_for_window(candidate, line, direction, contact_distance, speed,
                             target_offset, repeat_index, variant_index):
    """Create visible route context while retaining the recorded interaction."""
    if candidate["id"] == "scenario_2_window_06":
        # The focused forklift begins at the right and drives left. Forward
        # routes start at the left end and drive toward it; reverse routes
        # begin left of the forklift and continue away to the left boundary.
        if direction == "forward":
            start, end = min(line.length, repeat_index * 80.0), min(
                line.length, contact_distance + speed * (45 + repeat_index * 10)
            )
        else:
            left_start = line.project(Point(350 - repeat_index * 70,
                                            line.interpolate(contact_distance).y))
            start, end = min(left_start, line.length), line.length
        return min(start + variant_index * 45.0, max(0.0, end - 20.0)), end

    start = max(0.0, contact_distance - speed * target_offset)
    end = min(line.length, contact_distance + speed * 35)
    if candidate["id"] in LONG_LEFT_CONTEXT_WINDOWS:
        # Add substantial leftward context on the long top-side reference
        # routes instead of clipping each replay around the encounter point.
        if direction == "forward":
            start = max(0.0, contact_distance - 300.0 - repeat_index * 100.0)
        else:
            end = min(line.length, contact_distance + 400.0 + repeat_index * 100.0)
    elif repeat_index:
        # Distinct timing and visible starting position for unavoidable reused
        # close route geometry.
        shift = repeat_index * 60.0
        if direction == "forward":
            start = max(0.0, start - shift)
            end = min(line.length, end + shift * 0.5)
        else:
            start = max(0.0, start - shift * 0.5)
            end = min(line.length, end + shift)
    # Spacing variants along their directed route means that all ten AMRs can
    # be visually distinguished at the first replay frame, even on parallel
    # lanes sharing a reference-path endpoint.
    start = min(start + variant_index * 45.0, max(0.0, end - 20.0))
    return start, end


def separate_start_position(line, start, end, occupied_points):
    """Choose a visually distinct initial AMR location on the same route."""
    maximum = max(start, end - 20.0)
    candidate = start
    while candidate <= maximum + 1e-6:
        point = line.interpolate(candidate)
        if all(point.distance(other) >= MIN_START_SEPARATION_PX
               for other in occupied_points):
            return candidate, point
        candidate += MIN_START_SEPARATION_PX
    raise ValueError("Cannot separate Scenario 2 AMR starts by {} px".format(
        MIN_START_SEPARATION_PX
    ))


def ensure_minimum_path_length(line, start, end, contact_distance):
    """Keep the focused-forklift contact in a sufficiently long route segment."""
    start = min(start, contact_distance)
    end = max(end, contact_distance)
    missing = MIN_ROBOT_PATH_LENGTH_PX - (end - start)
    if missing > 0:
        extend_after = min(missing, line.length - end)
        end += extend_after
        start = max(0.0, start - (missing - extend_after))
    if end - start < MIN_ROBOT_PATH_LENGTH_PX - 1e-6:
        raise ValueError("Reference route cannot provide a {} px segment around focus contact".format(
            MIN_ROBOT_PATH_LENGTH_PX
        ))
    return start, end


def nominal_position(path, speed, frame_index):
    """Position after ``frame_index`` steps of nominal route following."""
    current = [float(path[0][0]), float(path[0][1])]
    waypoint = 1
    for _ in range(frame_index + 1):
        remaining = speed
        while remaining > 0 and waypoint < len(path):
            target = path[waypoint]
            distance = math.dist(current, target)
            if distance < 1e-9:
                waypoint += 1
                continue
            step = min(remaining, distance)
            current[0] += (target[0] - current[0]) * step / distance
            current[1] += (target[1] - current[1]) * step / distance
            remaining -= step
            if step >= distance - 1e-9:
                waypoint += 1
    return current


def focus_encounter_distance(robot_path, speed, replay_frames, focus_track_id):
    """Closest nominal AMR/focus-forklift distance during this replay window."""
    distances = []
    for frame_index, frame in enumerate(replay_frames):
        obj = next((item for item in frame["objects"]
                    if int(item["track_id"]) == int(focus_track_id)), None)
        if obj is not None:
            distances.append(math.dist(nominal_position(robot_path, speed, frame_index),
                                       centre(obj["polygon"])))
    return min(distances) if distances else math.inf


def retime_for_focus_encounter(line, waypoints, start, end, contact_distance,
                               speed, replay_frames, focus_track_id,
                               occupied_points):
    """Advance an AMR start until it reaches the focus during the window."""
    earliest = max(0.0, contact_distance - speed * (len(replay_frames) - 1))
    latest = min(contact_distance, line.length - MIN_ROBOT_PATH_LENGTH_PX)
    for respect_spacing in (True, False):
        for trial_start in range(int(math.ceil(earliest)), int(math.floor(latest)) + 1, 15):
            trial_start = float(trial_start)
            trial_end = max(contact_distance, trial_start + MIN_ROBOT_PATH_LENGTH_PX)
            if trial_end > line.length:
                continue
            point = line.interpolate(trial_start)
            if respect_spacing and any(point.distance(other) < MIN_START_SEPARATION_PX
                                       for other in occupied_points):
                continue
            trial_path = segment(waypoints, trial_start, trial_end)
            distance = focus_encounter_distance(
                trial_path, speed, replay_frames, focus_track_id
            )
            if distance <= MAX_FOCUS_ENCOUNTER_DISTANCE_PX:
                return trial_start, trial_end, point, trial_path, distance
    return start, end, line.interpolate(start), segment(waypoints, start, end), math.inf


def main():
    candidates = json.load(CANDIDATES.open())["windows"]
    references = json.load(REFERENCES.open())
    if len(references) < 10:
        raise ValueError("Scenario 2 needs at least 10 reference paths")
    paths = [(name, references[name]) for name in sorted(references, key=sort_key)]
    operating_area = Polygon(json.load(OPERATING_AREA.open())["polygon_image_px"])
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
            if not operating_area.covers(line):
                continue
            closest = min(
                ((line.distance(point), offset, line.project(point)) for offset, point in samples),
                key=lambda item: item[0],
            )
            if closest[0] <= RELEVANCE_DISTANCE_PX:
                relevant.append((closest, name, definition))
        if not relevant:
            raise ValueError("No Scenario 2 reference path is relevant to {}".format(candidate["id"]))
        relevant.sort(key=lambda item: item[0][0])
        selected = select_directed_paths(candidate, relevant, samples,
                                         candidate["window_end_frame"] -
                                         candidate["window_start_frame"] + 1)
        route_signatures = set()
        occupied_start_points = []
        for path_index, (closest, name, definition, direction, repeat_index) in enumerate(selected):
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
            start, end = robot_segment_for_window(
                candidate, line, direction, contact_distance, speed, target_offset,
                repeat_index, path_index
            )
            if repeat_index:
                # Keep repeated reference geometry visibly and temporally
                # distinct by choosing a different 200 px route window that
                # still contains the closest focused-forklift contact.
                low = max(0.0, contact_distance - MIN_ROBOT_PATH_LENGTH_PX)
                high = min(contact_distance, line.length - MIN_ROBOT_PATH_LENGTH_PX)
                if high > low:
                    start = min(high, low + repeat_index * 50.0)
                    end = start + MIN_ROBOT_PATH_LENGTH_PX
            start, end = ensure_minimum_path_length(
                line, start, end, contact_distance
            )
            try:
                start, start_point = separate_start_position(
                    line, start, end, occupied_start_points
                )
            except ValueError as error:
                # The local interaction crop can be too short to place a
                # distinct start. Extend along the same approved reference
                # route, rather than duplicating an AMR at another start.
                start, start_point = separate_start_position(
                    line, 0.0, line.length, occupied_start_points
                )
                end = max(end, min(line.length, start + 20.0))
            start, end = ensure_minimum_path_length(
                line, start, end, contact_distance
            )
            if end - start < 20:
                start, end = 0.0, line.length
            signature = (name, direction, round(start, 1), round(end, 1),
                         round(target_offset, 1))
            while signature in route_signatures:
                if end + 30.0 <= line.length:
                    end += 30.0
                elif start - 30.0 >= 0:
                    start -= 30.0
                else:
                    raise AssertionError("{} cannot make {} {} distinct".format(
                        candidate["id"], name, direction
                    ))
                signature = (name, direction, round(start, 1), round(end, 1),
                             round(target_offset, 1))
            robot_path = segment(waypoints, start, end)
            route_signatures.add(signature)
            if not operating_area.covers(LineString(robot_path)):
                raise ValueError("Generated {} route leaves the robot operating area".format(
                    candidate["id"]
                ))
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
            encounter_distance = focus_encounter_distance(
                robot_path, speed, replay_frames, candidate["focus_track_id"]
            )
            if encounter_distance > MAX_FOCUS_ENCOUNTER_DISTANCE_PX:
                start, end, start_point, robot_path, encounter_distance = retime_for_focus_encounter(
                    line, waypoints, start, end, contact_distance, speed,
                    replay_frames, candidate["focus_track_id"], occupied_start_points
                )
            if encounter_distance > MAX_FOCUS_ENCOUNTER_DISTANCE_PX:
                raise ValueError(
                    "{} path {} {} {} remains {:.1f} px from focused forklift during replay".format(
                        candidate["id"], path_index + 1, name, direction, encounter_distance
                    )
                )
            occupied_start_points.append(start_point)
            instances.append({
                "id": "scenario_2_window_{:02d}_path_{:02d}".format(window_index + 1, path_index + 1),
                "window_id": candidate["id"], "robot_path_id": "robot_path_{:02d}".format(path_index + 1),
                "robot_path_label": "Path {:02d} — {} ({}; timing {:02d}; variant {})".format(path_index + 1, name, direction, target_offset, repeat_index + 1),
                "title": "{}; track {}; {}".format(candidate["id"], candidate["focus_track_id"], kind),
                "focus_track_id": candidate["focus_track_id"], "coverage": kind,
                "desired_behavior": {"expected_action": kind, "specification_rule": rule,
                                     "minimum_recorded_path_distance_px": round(minimum_distance, 1),
                                     "minimum_nominal_encounter_distance_px": round(encounter_distance, 1),
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
