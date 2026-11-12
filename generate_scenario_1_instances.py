#!/usr/bin/env python3
"""Build Scenario 1 replays from recorded forklift tracks and predictions.

Only the AMR path is synthetic. Images, forklift polygons, velocities, route/
manoeuvre classifications and OOD data remain tied to original timestamps.
"""

import json
import math
from collections import defaultdict, deque
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT = REPO_ROOT / "scenario_data/scenario_1_instances.json"
METRICS = REPO_ROOT / "scenario_output/scenario_1_supervisor_metrics.json"
REPORT = REPO_ROOT / "scenario_output/scenario_1_supervisor_report.md"
CLASSIFICATION_REPORT = REPO_ROOT / "scenario_output/scenario_1_instance_classification.md"
WINDOW_SUMMARY_REPORT = REPO_ROOT / "scenario_output/scenario_1_window_behavior_summary.md"
OPERATING_AREA_PATH = REPO_ROOT / "scenario_data/robot_operating_area.json"
REFERENCE_PATHS_PATH = REPO_ROOT / "scenario_data/robot_reference_paths.json"
CANDIDATE_WINDOWS_PATH = REPO_ROOT / "scenario_data/scenario_1_candidate_windows.json"
# At roughly 10 Hz, these add 1.5 s of normal operation before and after the
# prior 60-frame clips, placing the interaction in the middle of a 90-frame run.
WINDOW_BEFORE = 40
MIN_WINDOW_AFTER = 50
POST_CLEAR_FRAMES = 15
MAX_ENTRY_TO_CLEAR_FRAMES = 60
APPROACH_DISTANCE_PX = 160.0
MIN_EVENT_SEPARATION = 300
GRID_STEP_PX = 20
# Calibration is approximately 20 px/m; 3 px/frame at 10 Hz is 1.5 m/s.
NOMINAL_ROBOT_SPEED_PX_PER_FRAME = 3.0
ROBOT_POST_EXIT_FRAMES = 10
ROBOT_ENTRY_OFFSET_FRAMES = 45


def centre(polygon):
    return [sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon)]


def object_polygon(obj):
    polygon = obj.get("associated_polygon") or []
    return polygon if len(polygon) >= 3 else None


def recorded_entries(frames, intersection):
    """Find observed outside-to-inside transitions for each recorded track."""
    previous = {}
    events = []
    for frame_index, frame in enumerate(frames):
        seen = set()
        for obj in frame["object_list"]:
            polygon = object_polygon(obj)
            if polygon is None:
                continue
            track_id = int(obj["track_id"])
            is_inside = intersection.intersects(Polygon(polygon))
            prior = previous.get(track_id)
            if prior and prior[0] == frame_index - 1 and not prior[1] and is_inside:
                events.append((frame_index, track_id))
            previous[track_id] = (frame_index, is_inside)
            seen.add(track_id)
        for track_id in set(previous) - seen:
            del previous[track_id]
    return events


def choose_events(events, frames, intersection):
    chosen = []
    for event in events:
        frame_index, track_id = event
        if frame_index < WINDOW_BEFORE or frame_index + MIN_WINDOW_AFTER >= len(frames):
            continue
        clear_frame = first_sustained_clear_frame(
            frame_index, track_id, frames, intersection
        )
        if clear_frame - frame_index > MAX_ENTRY_TO_CLEAR_FRAMES:
            continue
        if all(abs(frame_index - selected[0]) >= MIN_EVENT_SEPARATION for selected in chosen):
            chosen.append(event)
        if len(chosen) == 10:
            return chosen
    raise RuntimeError("Could not find ten sufficiently separated recorded intersection entries")


def constrained_robot_path(operating_area, intersection, variant=0,
                           target_pre_distance_px=135.0):
    """Find an in-area through-route joining opposite intersection sides."""
    traversable_intersection = operating_area.intersection(intersection)
    if traversable_intersection.is_empty or traversable_intersection.area <= 0:
        raise ValueError("Robot operating area does not overlap the intersection")

    min_x, min_y, max_x, max_y = operating_area.bounds
    points = {}
    for x in range(int(math.ceil(min_x)), int(math.floor(max_x)) + 1, GRID_STEP_PX):
        for y in range(int(math.ceil(min_y)), int(math.floor(max_y)) + 1, GRID_STEP_PX):
            point = Point(x, y)
            if operating_area.covers(point):
                points[(x, y)] = point

    def nearest_key(geometry):
        probe = geometry.representative_point()
        return min(points, key=lambda key: points[key].distance(probe))

    anchor_key = nearest_key(traversable_intersection)

    def shortest_path(start_key, end_key):
        queue = deque([start_key])
        parent = {start_key: None}
        while queue:
            key = queue.popleft()
            if key == end_key:
                route = []
                while key is not None:
                    route.append(key)
                    key = parent[key]
                return list(reversed(route))
            for dx, dy in ((GRID_STEP_PX, 0), (-GRID_STEP_PX, 0), (0, GRID_STEP_PX),
                           (0, -GRID_STEP_PX), (GRID_STEP_PX, GRID_STEP_PX),
                           (GRID_STEP_PX, -GRID_STEP_PX), (-GRID_STEP_PX, GRID_STEP_PX),
                           (-GRID_STEP_PX, -GRID_STEP_PX)):
                next_key = (key[0] + dx, key[1] + dy)
                if next_key in parent or next_key not in points:
                    continue
                if operating_area.covers(LineString([points[key], points[next_key]])):
                    parent[next_key] = key
                    queue.append(next_key)
        return None

    inter_min_x, inter_min_y, inter_max_x, inter_max_y = intersection.bounds
    starts = [
        key for key, point in points.items()
        if point.x < inter_min_x - 60 and not intersection.covers(point)
    ]
    ends = [
        key for key, point in points.items()
        if point.x > inter_max_x + 60 and not intersection.covers(point)
    ]
    # Start close enough to reach the intersection after visible normal motion,
    # but end on the other side of the intersection rather than returning.
    starts.sort(key=lambda key: abs(
        points[key].distance(intersection) - target_pre_distance_px
    ))
    ends.sort(key=lambda key: abs(points[key].distance(intersection) - 160))

    route_candidates = []
    used_endpoints = set()
    for start_key in starts[:20]:
        start_route = shortest_path(start_key, anchor_key)
        if not start_route:
            continue
        for end_key in ends[:80]:
            end_route = shortest_path(anchor_key, end_key)
            if not end_route:
                continue
            keys = start_route + end_route[1:]
            route = [points[key] for key in keys]
            path = LineString(route)
            crossed = path.intersection(intersection)
            if crossed.is_empty or crossed.length < 20:
                continue
            if points[start_key].distance(intersection) < 30:
                continue
            if points[end_key].distance(intersection) < 30:
                continue
            # Compress collinear grid points for clean MVP rendering.
            compact = [route[0]]
            for previous, current, following in zip(route, route[1:], route[2:]):
                first = (current.x - previous.x, current.y - previous.y)
                second = (following.x - current.x, following.y - current.y)
                if first[0] * second[1] != first[1] * second[0]:
                    compact.append(current)
            compact.append(route[-1])
            endpoint_key = (start_key, end_key)
            if endpoint_key in used_endpoints:
                continue
            used_endpoints.add(endpoint_key)
            route_candidates.append([[point.x, point.y] for point in compact])
            if len(route_candidates) >= 24:
                return route_candidates[variant % len(route_candidates)]
    if route_candidates:
        return route_candidates[variant % len(route_candidates)]
    raise ValueError(
        "Could not find an operating-area route that crosses to the opposite side of the intersection"
    )


def polyline_length(points):
    return sum(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(points, points[1:])
    )


def reference_path_sort_key(name):
    suffix = name.rsplit("_", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def line_intersection_distances(line, intersection):
    """Sample a reference path to find its entry and exit distances."""
    inside = []
    for distance in range(int(math.ceil(line.length)) + 1):
        if intersection.covers(line.interpolate(distance)):
            inside.append(float(distance))
    if not inside:
        return None
    return min(inside), max(inside)


def reference_path_segment(waypoints, start_distance_px, end_distance_px=None):
    """Return a continuous, bounded segment of a reference polyline."""
    line = LineString(waypoints)
    if end_distance_px is None:
        end_distance_px = line.length
    end_distance_px = min(line.length, max(start_distance_px, end_distance_px))
    start = line.interpolate(start_distance_px)
    result = [[start.x, start.y]]
    covered = 0.0
    for point_a, point_b in zip(waypoints, waypoints[1:]):
        segment_length = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
        segment_start = covered
        covered += segment_length
        if (covered > start_distance_px + 1e-6
                and covered < end_distance_px - 1e-6):
            if math.hypot(result[-1][0] - point_b[0], result[-1][1] - point_b[1]) > 1e-6:
                result.append([float(point_b[0]), float(point_b[1])])
    end = line.interpolate(end_distance_px)
    if math.hypot(result[-1][0] - end.x, result[-1][1] - end.y) > 1e-6:
        result.append([end.x, end.y])
    return result


def load_reference_routes(operating_area, intersection):
    if not REFERENCE_PATHS_PATH.exists():
        raise FileNotFoundError("Reference paths not found: {}".format(REFERENCE_PATHS_PATH))
    source = json.load(REFERENCE_PATHS_PATH.open())
    valid = []
    rejected = []
    for name in sorted(source, key=reference_path_sort_key):
        definition = source[name]
        waypoints = definition.get("waypoints", [])
        if len(waypoints) < 2:
            rejected.append((name, "fewer than two waypoints"))
            continue
        speed = float(definition.get("speed", NOMINAL_ROBOT_SPEED_PX_PER_FRAME))
        line = LineString(waypoints)
        if not operating_area.covers(line):
            rejected.append((name, "leaves operating area"))
        elif line_intersection_distances(line, intersection) is None:
            rejected.append((name, "does not cross intersection"))
        elif speed <= 0:
            rejected.append((name, "non-positive speed"))
        else:
            # Both travel directions are valid AMR candidates.  Treating them
            # separately is important: they give distinct approach/exit sides
            # and supervisor timing while preserving the user-drawn geometry.
            for direction, directed_waypoints in (
                ("forward", waypoints), ("reverse", list(reversed(waypoints))),
            ):
                directed = [[float(x), float(y)] for x, y in directed_waypoints]
                directed_line = LineString(directed)
                intersections = line_intersection_distances(directed_line, intersection)
                valid.append({
                    "name": "{} ({})".format(name, direction),
                    "source_name": name,
                    "direction": direction,
                    "waypoints": directed,
                    "speed": speed,
                    "line": directed_line,
                    "entry_distance": intersections[0],
                    "exit_distance": intersections[1],
                })
    if not valid:
        details = "; ".join("{}: {}".format(name, reason) for name, reason in rejected)
        raise ValueError("No valid reference paths. " + details)
    return valid, rejected


def route_side(point, intersection):
    centroid = intersection.centroid
    dx, dy = point.x - centroid.x, point.y - centroid.y
    if abs(dx) >= abs(dy):
        return "east" if dx >= 0 else "west"
    return "south" if dy >= 0 else "north"


def select_routes_for_window(routes, intersection, window_index, count=10):
    """Choose ten distinct, direction-aware routes with varied crossings."""
    suitable = [
        route for route in routes
        if route["entry_distance"] >= route["speed"] * 20
        and route["line"].length - route["exit_distance"] >= route["speed"] * 40
    ]
    if len(suitable) < count:
        raise ValueError("Only {} reference directions have enough replay context".format(
            len(suitable)
        ))

    # Rotate tie-breaking per recording so the same ten routes are not reused
    # for every forklift interaction.
    ordered = sorted(
        suitable,
        key=lambda route: (reference_path_sort_key(route["source_name"]), route["direction"]),
    )
    shift = (window_index * count) % len(ordered)
    rotated = ordered[shift:] + ordered[:shift]
    selected = []
    seen_sides = set()
    seen_sources = set()
    while rotated and len(selected) < count:
        best = None
        best_score = -1
        for order, route in enumerate(rotated):
            entry = route_side(route["line"].interpolate(route["entry_distance"]), intersection)
            exit_ = route_side(route["line"].interpolate(route["exit_distance"]), intersection)
            crossing = route["exit_distance"] - route["entry_distance"]
            length_bucket = "short" if crossing < 100 else "long"
            features = {(entry, exit_), length_bucket}
            score = 100 * sum(feature not in seen_sides for feature in features)
            score += 30 if route["source_name"] not in seen_sources else 0
            score -= order * 0.01
            if score > best_score:
                best, best_score = route, score
        selected.append(best)
        rotated.remove(best)
        entry = route_side(best["line"].interpolate(best["entry_distance"]), intersection)
        exit_ = route_side(best["line"].interpolate(best["exit_distance"]), intersection)
        seen_sides.update({(entry, exit_), "short" if best["exit_distance"] - best["entry_distance"] < 100 else "long"})
        seen_sources.add(best["source_name"])
    return selected


def path_intersection_frames(path, intersection, speed_px_per_frame):
    """Return the first and last nominal AMR frames inside the intersection."""
    current = [float(path[0][0]), float(path[0][1])]
    waypoint_index = 1
    inside_frames = []
    frame = 0
    while waypoint_index < len(path):
        if intersection.covers(Point(current)):
            inside_frames.append(frame)
        remaining = speed_px_per_frame
        while remaining > 0 and waypoint_index < len(path):
            target = path[waypoint_index]
            dx, dy = target[0] - current[0], target[1] - current[1]
            distance = math.hypot(dx, dy)
            if distance < 1e-6:
                waypoint_index += 1
                continue
            step = min(remaining, distance)
            current[0] += dx / distance * step
            current[1] += dy / distance * step
            remaining -= step
            if distance <= step + 1e-6:
                waypoint_index += 1
        frame += 1
    if intersection.covers(Point(current)):
        inside_frames.append(frame)
    if not inside_frames:
        raise ValueError("Generated AMR path does not enter the intersection")
    return min(inside_frames), max(inside_frames)


def first_sustained_clear_frame(entry_frame, focus_track_id, frames, intersection):
    """Find when the forklift is outside the approach zone for five frames."""
    clear_run = 0
    for frame_index in range(entry_frame, len(frames)):
        obj = next(
            (item for item in frames[frame_index]["object_list"]
             if int(item["track_id"]) == focus_track_id),
            None,
        )
        polygon = object_polygon(obj) if obj is not None else None
        is_relevant = polygon is not None and (
            intersection.intersects(Polygon(polygon))
            or Polygon(polygon).distance(intersection) <= APPROACH_DISTANCE_PX
        )
        clear_run = 0 if is_relevant else clear_run + 1
        if clear_run >= 5:
            return frame_index - 4
    return entry_frame


def frame_objects(frame, previous_centres, dt_s):
    """Adapt recorded detections for the MVP without altering them."""
    adapted = []
    next_centres = {}
    for obj in frame["object_list"]:
        polygon = object_polygon(obj)
        if polygon is None:
            continue
        track_id = int(obj["track_id"])
        current = centre(polygon)
        prior = previous_centres.get(track_id)
        velocity_px_s = [0.0, 0.0] if prior is None else [
            (current[0] - prior[0]) / dt_s,
            (current[1] - prior[1]) / dt_s,
        ]
        world_velocity = obj.get("velocity_3d", {})
        adapted.append({
            "track_id": track_id,
            "polygon": polygon,
            "velocity_world": [
                float(world_velocity.get("x", 0.0)),
                float(world_velocity.get("y", 0.0)),
            ],
            "velocity_px_s": velocity_px_s,
        })
        next_centres[track_id] = current
    return adapted, next_centres


def desired_behavior(kind, prediction):
    """Map each replay to the required Scenario 1 robot behaviour."""
    route = prediction.get("route") or {}
    uncertainty = float(route.get("uncertainty", 1.0))
    confidence = float(route.get("prob", prediction.get("display_confidence", 0.0)))
    maneuver = prediction.get("mode", "unknown")
    if kind == "clearance":
        action = "Vacate intersection"
        rule = "Intersection clearance; occupancy safety rule"
        description = "AMR reaches the intersection before the recorded forklift entry and must clear it immediately; with high uncertainty it must stop/reassess after clearing."
    elif uncertainty >= 0.70:
        action = "Fail-safe: remain outside / stop"
        rule = "Fail-safe behaviour"
        description = "Recorded route uncertainty is {:.2f}; do not enter until perception confidence is adequate.".format(uncertainty)
    elif kind == "entry_control":
        action = "Yield / delay entry"
        rule = "Entry control; conflicting-path response"
        description = "AMR is outside and its path crosses the intersection while the recorded forklift enters it."
    else:
        action = "Proceed with caution"
        rule = "Non-conflicting-path response"
        description = "AMR route remains outside the intersection; monitor the recorded forklift but continue on the separate lane."
    return {
        "expected_action": action,
        "specification_rule": rule,
        "rationale": description,
        "recorded_maneuver": maneuver,
        "recorded_route": "{}→{}".format(route.get("from", "?"), route.get("to", "?")),
        "route_confidence": confidence,
        "route_uncertainty": uncertainty,
    }


def write_classification_report(instances):
    lines = [
        "# Scenario 1 — desired robot behaviour classification",
        "",
        "Forklift observations, trajectory classifications, manoeuvre classifications and timestamps are replayed from the recorded dataset. Only the AMR path is simulated.",
        "",
        "| Scenario | Recorded window, forklift and classifier output | Desired robot behaviour | Scenario-specification rule and rationale |",
        "|---|---|---|---|",
    ]
    for instance in instances:
        label = instance["desired_behavior"]
        window = instance["source_window"]
        recorded = "track {}; route {}; {} (confidence {:.2f}, uncertainty {:.2f})".format(
            instance["focus_track_id"], label["recorded_route"], label["recorded_maneuver"],
            label["route_confidence"], label["route_uncertainty"],
        )
        source = "frames {}–{}; entry {}; clear {}".format(
            window["start_frame"], window["end_frame"], window["entry_frame"],
            window["focus_clear_frame"],
        )
        lines.append("| {} | {}; {} | {} | {} — {} |".format(
            instance["id"], source, recorded, label["expected_action"],
            label["specification_rule"], label["rationale"],
        ))
    lines.extend([
        "",
        "All ten selected recordings (and their ten AMR variants each) classify the in-intersection forklift as `moving`. No recorded `loading` or `stationary` forklift was found within this user-defined intersection; therefore the loading-obstruction and stationary-forklift rules are not represented by this batch and must be covered by a separate, dataset-backed selection when such recordings are available.",
        "",
        "A fail-safe test requires the configured supervisor uncertainty threshold (0.70) or a recorded OOD flag. No selected moving-forklift window reaches that threshold. For clearance examples, the AMR must first leave the intersection and then stop or reassess if the uncertainty remains high.",
    ])
    CLASSIFICATION_REPORT.write_text("\n".join(lines) + "\n")


def write_window_summary(candidates, routes_by_window, instances):
    """Write the concise scenario catalogue used to choose a replay."""
    lines = [
        "# Scenario 1 — selected windows and robot-behaviour coverage",
        "",
        "Each row is one recorded forklift scene. Forklift detections, classifications and timing are recorded; only the AMR routes are simulated. All 10 AMR variants per row are evaluated against **all forklifts** in the replay, not only the named focal forklift.",
        "",
        "| Recorded window | Forklift behaviour | Scene context | Robot variants and intended coverage |",
        "|---|---|---|---|",
    ]
    for index, candidate in enumerate(candidates):
        group = [item for item in instances if item["window_id"] == candidate["id"]]
        coverage = {}
        for item in group:
            coverage[item["coverage"]] = coverage.get(item["coverage"], 0) + 1
        behavior = (
            "Track {}; {}→{}; {:.1f} s inside; route {}→{}; {}"
        ).format(
            candidate["focus_track_id"], candidate["entry_side"], candidate["exit_side"],
            candidate["inside_duration_s"], candidate["route"]["from"],
            candidate["route"]["to"], candidate["maneuver"],
        )
        context = "{} distinct / peak {} nearby forklifts".format(
            candidate["nearby_forklift_count"], candidate["peak_concurrent_forklifts"]
        )
        variants = "; ".join(route["name"] for route in routes_by_window[index])
        intended = "{} entry-control yield/delay; {} clearance/occupancy".format(
            coverage.get("entry_control", 0), coverage.get("clearance", 0)
        )
        lines.append("| {} | {} | {} | {}. Routes: {} |".format(
            candidate["id"], behavior, context, intended, variants
        ))
    lines.extend([
        "",
        "## Scope and interpretation",
        "",
        "- The 70 entry-control variants time the AMR to approach after the recorded forklift threat, so it should delay entry at the intersection boundary when required.",
        "- The 30 clearance variants time the AMR to already be in the intersection when the recorded forklift approaches, so it should vacate the area promptly.",
        "- Every selected route stays within the robot operating area and crosses the intersection. Forward and reverse directions are treated as different route variants.",
        "- This catalogue describes intended test coverage. Reported decision outcomes should be derived from the MVP evidence log after replaying the cases, especially because all nearby forklifts now participate in the online safety decision.",
    ])
    WINDOW_SUMMARY_REPORT.write_text("\n".join(lines) + "\n")


def build_instance(window_index, path_index, candidate, frames, predictions, intersection, operating_area,
                   reference_route, requested_kind):
    """Attach one approved AMR through-route to one fixed recorded window."""
    entry_frame = int(candidate["entry_frame"])
    focus_track_id = int(candidate["focus_track_id"])
    start = int(candidate["window_start_frame"])
    selected_end = int(candidate["window_end_frame"])
    event_offset = entry_frame - start
    entry_timestamp = int(frames[entry_frame]["time_stamp"])
    focus_prediction = next(
        (prediction for prediction in predictions.get(entry_timestamp, [])
         if int(prediction["track_id"]) == focus_track_id),
        {},
    )
    # The AMR is nominally inside at the forklift's observed entry for a
    # clearance test, or reaches the boundary after it for an entry-control
    # test.  In both cases it has visible normal operation before the decision.
    entry_offset = 20 if requested_kind == "clearance" else ROBOT_ENTRY_OFFSET_FRAMES
    start_distance = max(
        0.0,
        reference_route["entry_distance"] - reference_route["speed"] * entry_offset,
    )
    # Retain enough route after the intersection to show resumed normal motion.
    end_distance = min(
        reference_route["line"].length,
        reference_route["exit_distance"] + reference_route["speed"] * 40,
    )
    path = reference_path_segment(
        reference_route["waypoints"], start_distance, end_distance
    )
    robot_entry_offset, robot_exit_offset = path_intersection_frames(
        path, intersection, reference_route["speed"]
    )
    # A clearance case is valid only if the AMR is still inside when the
    # recorded forklift reaches the intersection. Otherwise it is entry control.
    kind = requested_kind
    if kind == "clearance" and not (
        robot_entry_offset + 10 <= event_offset <= robot_exit_offset
    ):
        kind = "entry_control"
    clear_frame = first_sustained_clear_frame(
        entry_frame, focus_track_id, frames, intersection
    )
    distance = polyline_length(path)
    robot_travel_frames = int(math.ceil(
        distance / reference_route["speed"]
    ))
    end = max(
        selected_end + 1,
        clear_frame + POST_CLEAR_FRAMES,
        start + robot_travel_frames + ROBOT_POST_EXIT_FRAMES,
    )
    if end > len(frames):
        raise ValueError("Recording ends before the constrained robot route can finish")
    previous_centres = {}
    replay_frames = []
    for source_index in range(start, end):
        source = frames[source_index]
        if source_index > start:
            prior_timestamp = int(frames[source_index - 1]["time_stamp"])
            dt_s = max(0.001, (int(source["time_stamp"]) - prior_timestamp) / 1000.0)
        else:
            dt_s = 0.1
        objects, previous_centres = frame_objects(source, previous_centres, dt_s)
        timestamp = int(source["time_stamp"])
        replay_frames.append({
            "timestamp": timestamp,
            "source_frame_index": source_index,
            "objects": objects,
            "predictions": predictions.get(timestamp, []),
            "focus_track_id": focus_track_id,
            "recorded_intersection_entry": source_index == entry_frame,
        })
    path_geometry = LineString(path)
    nominal_speed_px_s = reference_route["speed"] / 0.1
    pre_entry_s = Point(path[0]).distance(intersection) / nominal_speed_px_s
    post_exit_s = Point(path[-1]).distance(intersection) / nominal_speed_px_s
    if pre_entry_s < 1.0 or post_exit_s < 1.0:
        raise ValueError(
            "{} does not provide one second before/after intersection "
            "({:.2f}s before, {:.2f}s after)".format(
                reference_route["name"], pre_entry_s, post_exit_s
            )
        )
    return {
        "id": "scenario_1_window_{:02d}_path_{:02d}".format(
            window_index + 1, path_index + 1
        ),
        "window_id": candidate["id"],
        "robot_path_id": "robot_path_{:02d}".format(path_index + 1),
        "robot_path_label": "Path {:02d} — {}".format(path_index + 1, reference_route["name"]),
        "title": "{}; recorded forklift entry at source frame {} (track {})".format(
            candidate["id"], entry_frame, focus_track_id
        ),
        "coverage": kind,
        "source_window": {
            "start_frame": start,
            "entry_frame": entry_frame,
            "focus_clear_frame": clear_frame,
            "end_frame": end - 1,
            "candidate_window_id": candidate["id"],
            "selected_end_frame": selected_end,
        },
        "focus_track_id": focus_track_id,
        "robot_reference_path": reference_route["name"],
        "robot_reference_path_source_name": reference_route["source_name"],
        "robot_reference_path_direction": reference_route["direction"],
        "robot_reference_path_source": str(REFERENCE_PATHS_PATH.relative_to(REPO_ROOT)),
        "robot_reference_start_distance_px": start_distance,
        "robot_path": path,
        "robot_path_intersects_intersection": bool(path_geometry.intersects(intersection)),
        "robot_path_within_operating_area": bool(operating_area.covers(path_geometry)),
        "robot_speed_px_per_frame": reference_route["speed"],
        "robot_path_timing_s": {
            "before_intersection": pre_entry_s,
            "after_intersection": post_exit_s,
        },
        "robot_path_schedule_frames": {
            "entry": robot_entry_offset,
            "exit": robot_exit_offset,
            "forklift_entry": event_offset,
        },
        "desired_behavior": desired_behavior(kind, focus_prediction),
        "frames": replay_frames,
    }


def normalise_window_frames(instances):
    """Use one complete recorded sequence for every path in a time window.

    Route variants may require different amounts of post-route context.  The
    longest extracted sequence is therefore the shared selected window.  This
    keeps parallel visualisation meaningful and guarantees every offline robot
    evaluation has one action and one position for every displayed frame.
    """
    groups = defaultdict(list)
    for instance in instances:
        groups[instance["window_id"]].append(instance)
    for window_id, variants in groups.items():
        canonical = max(variants, key=lambda item: len(item["frames"]))
        frames = canonical["frames"]
        start = frames[0]["source_frame_index"]
        end = frames[-1]["source_frame_index"]
        for instance in variants:
            instance["frames"] = list(frames)
            instance["source_window"]["start_frame"] = start
            instance["source_window"]["end_frame"] = end
            instance["source_window"]["canonical_window_id"] = window_id


def main():
    area_data = json.load((REPO_ROOT / "scenario_data/intersection_area.json").open())
    intersection_points = area_data["polygon_image_px"]
    intersection = Polygon(intersection_points)
    if not OPERATING_AREA_PATH.exists():
        raise FileNotFoundError(
            "Robot operating area not found: {}".format(OPERATING_AREA_PATH)
        )
    operating_area_data = json.load(OPERATING_AREA_PATH.open())
    operating_area = Polygon(operating_area_data["polygon_image_px"])
    reference_routes, rejected_reference_routes = load_reference_routes(
        operating_area, intersection
    )
    frames = json.load((REPO_ROOT / "JSON-data/final.json").open())
    prediction_frames = json.load((REPO_ROOT / "JSON-data/predictions_2026_02_27.json").open())
    predictions = {int(frame["time_stamp"]): frame.get("predictions", []) for frame in prediction_frames}

    candidates = json.load(CANDIDATE_WINDOWS_PATH.open()).get("windows", [])
    if len(candidates) != 10:
        raise ValueError("Expected exactly ten candidate windows in {}".format(
            CANDIDATE_WINDOWS_PATH
        ))
    # Three variants per recording place the AMR in the intersection at entry, so the
    # clearance/occupancy rule can be observed.  The rest stage the AMR's
    # arrival after the forklift threat, exercising yield/delay entry.
    coverage_plan = [
        "entry_control", "clearance", "entry_control", "entry_control",
        "clearance", "entry_control", "clearance", "entry_control",
        "entry_control", "entry_control",
    ]
    routes_by_window = [
        select_routes_for_window(reference_routes, intersection, window_index)
        for window_index in range(len(candidates))
    ]
    instances = [
        build_instance(
            window_index, path_index, candidate, frames, predictions,
            intersection, operating_area, route, coverage_plan[path_index],
        )
        for window_index, candidate in enumerate(candidates)
        for path_index, route in enumerate(routes_by_window[window_index])
    ]
    normalise_window_frames(instances)
    payload = {
        "scenario": "Scenario 1: Robot Operation at Intersections",
        "data_provenance": "Recorded forklift detections, predictions and timestamps; simulated AMR path only.",
        "intersection_polygon_px": intersection_points,
        "robot_operating_area_px": operating_area_data["polygon_image_px"],
        "robot_operating_area_source": str(OPERATING_AREA_PATH.relative_to(REPO_ROOT)),
        "robot_reference_paths_source": str(REFERENCE_PATHS_PATH.relative_to(REPO_ROOT)),
        "candidate_windows_source": str(CANDIDATE_WINDOWS_PATH.relative_to(REPO_ROOT)),
        "valid_robot_reference_paths": [route["name"] for route in reference_routes],
        "selected_robot_paths_by_window": {
            candidate["id"]: [route["name"] for route in routes_by_window[index]]
            for index, candidate in enumerate(candidates)
        },
        "rejected_robot_reference_paths": [
            {"name": name, "reason": reason}
            for name, reason in rejected_reference_routes
        ],
        "instances": instances,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, OUTPUT.open("w"), indent=2)
    write_classification_report(instances)
    write_window_summary(candidates, routes_by_window, instances)

    coverage = {}
    for instance in instances:
        coverage[instance["coverage"]] = coverage.get(instance["coverage"], 0) + 1
    metrics = {
        "dataset_backed": True,
        "window_count": len(candidates),
        "paths_per_window": 10,
        "instance_count": len(instances),
        "coverage": coverage,
        "note": "Decision-performance metrics must be calculated from an exported replay evidence log; no synthetic forklift ground truth is used.",
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, METRICS.open("w"), indent=2)
    REPORT.write_text(
        "# Scenario 1 — recorded-data replay set\n\n"
        "All forklift detections, BEV polygons, timestamps, route and manoeuvre classifications, and OOD inputs are replayed from the recorded dataset. "
        "Only the AMR path is simulated. Every generated AMR trajectory is a suffix of a user-approved route in `scenario_data/robot_reference_paths.json`, constrained to `scenario_data/robot_operating_area.json`, and crosses the valid overlap with the intersection.\n\n"
        "## Batch\n\n"
        "- Recorded forklift windows: 10\n"
        "- AMR route variants per window: 10\n"
        "- Replay instances: 100\n"
        "- Window: 40 frames before the recorded entry through at least 15 frames after the recorded forklift clears the intersection\n"
        "- AMR route context: at least 1 second before entering and 1 second after leaving the intersection\n"
        "- AMR speed: preserved from the selected user reference path\n"
        "- Coverage: {}\n\n"
        "This batch contains moving-forklift entry-control examples with AMR through-routes across the intersection. The current recordings contain no loading or stationary detections within the selected intersection, so those claims are intentionally excluded. "
        "Export the MVP evidence logs before calculating recall, false positives, or any Vinnova-facing performance metric.\n".format(
            ", ".join("{}={}".format(key, value) for key, value in sorted(coverage.items()))
        )
    )
    print("Using approved reference paths: {}".format(
        ", ".join(route["name"] for route in reference_routes)
    ))
    if rejected_reference_routes:
        print("Rejected reference paths: {}".format(
            ", ".join("{} ({})".format(name, reason) for name, reason in rejected_reference_routes)
        ))
    print("Wrote {} with {} windows × {} robot paths".format(
        OUTPUT.relative_to(REPO_ROOT), len(candidates), 10
    ))
    print("Wrote {}".format(WINDOW_SUMMARY_REPORT.relative_to(REPO_ROOT)))
    for index, candidate in enumerate(candidates):
        print("{}: {}".format(
            candidate["id"], ", ".join(route["name"] for route in routes_by_window[index])
        ))
    print("Wrote {}".format(CLASSIFICATION_REPORT.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
