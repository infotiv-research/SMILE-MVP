#!/usr/bin/env python3
"""Measure recorded-data Scenario 1 replays with the intersection supervisor."""

import json
import math
from collections import Counter
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon

from scenario_supervisor import evaluate_scenario_1


ROOT = Path(__file__).resolve().parent
REPLAYS = ROOT / "JSON-data/scenario_1_instances.json"
SUMMARY = ROOT / "report_outputs/scenario_1_window_behavior_summary.md"
RESULTS = ROOT / "report_outputs/scenario_1_measured_outcomes.json"


def move(state, intersection):
    """Apply the same STOP/YIELD/CLEAR movement policy as the viewer."""
    if state["action"] == "STOP" or state["completed"]:
        return
    speed = state["speed"]
    if state["action"] == "SLOW":
        speed *= 0.5
    elif state["action"] == "CLEAR_INTERSECTION":
        speed *= 1.5
    if state["action"] == "YIELD":
        distance = Point(state["xy"]).distance(intersection)
        if distance <= 3.0:
            return
        speed = min(speed, max(0.0, distance - 3.0))
    while speed > 0 and state["waypoint"] < len(state["path"]):
        target = state["path"][state["waypoint"]]
        dx, dy = target[0] - state["xy"][0], target[1] - state["xy"][1]
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            state["waypoint"] += 1
        elif distance <= speed:
            state["xy"] = list(target)
            state["waypoint"] += 1
            speed -= distance
        else:
            state["xy"][0] += dx / distance * speed
            state["xy"][1] += dy / distance * speed
            speed = 0
    state["completed"] = state["waypoint"] >= len(state["path"])


def route_enters(state, intersection):
    remaining = [state["xy"]] + state["path"][state["waypoint"]:]
    return len(remaining) >= 2 and intersection.intersects(LineString(remaining))


def time_to_intersection(state, intersection):
    if not route_enters(state, intersection):
        return math.inf
    return Point(state["xy"]).distance(intersection) / (state["speed"] / 0.1)


def evaluate_instance(instance, intersection):
    state = {
        "path": [list(point) for point in instance["robot_path"]],
        "xy": list(instance["robot_path"][0]),
        "speed": float(instance["robot_speed_px_per_frame"]),
        "waypoint": 1, "action": "GO", "completed": False,
    }
    actions, reasons = [], []
    yielded_at_boundary = False
    clear_while_inside = False
    for frame in instance["frames"]:
        move(state, intersection)
        predictions = {int(item["track_id"]): item for item in frame.get("predictions", [])}
        forklifts = []
        for obj in frame["objects"]:
            pred = predictions.get(int(obj["track_id"]), {})
            forklifts.append({
                "track_id": obj["track_id"], "polygon": obj["polygon"],
                "velocity_px_s": obj.get("velocity_px_s", [0.0, 0.0]),
                "maneuver": pred.get("mode", "moving"),
            })
        result = evaluate_scenario_1(
            state["xy"], forklifts, list(intersection.exterior.coords)[:-1],
            robot_path_intersects_intersection=route_enters(state, intersection),
            robot_entry_time_s=time_to_intersection(state, intersection),
        )
        state["action"] = result["action"]
        actions.append(result["action"])
        reasons.append(result["reason"])
        if result["action"] == "YIELD" and Point(state["xy"]).distance(intersection) <= 3.1:
            yielded_at_boundary = True
        if result["action"] == "CLEAR_INTERSECTION" and intersection.covers(Point(state["xy"])):
            clear_while_inside = True
    counts = Counter(actions)
    expected = instance["coverage"]
    passed = yielded_at_boundary if expected == "entry_control" else clear_while_inside
    return {
        "id": instance["id"], "window_id": instance["window_id"],
        "expected_coverage": expected, "action_counts": dict(counts),
        "reasons": dict(Counter(reasons)), "yielded_at_boundary": yielded_at_boundary,
        "clear_command_while_inside": clear_while_inside, "completed_route": state["completed"],
        "pass": passed,
    }


def main():
    payload = json.load(REPLAYS.open())
    intersection = Polygon(payload["intersection_polygon_px"])
    outcomes = [evaluate_instance(instance, intersection) for instance in payload["instances"]]
    json.dump({"method": "Online-equivalent intersection supervisor replay; OOD/TTC layers excluded", "outcomes": outcomes}, RESULTS.open("w"), indent=2)
    grouped = {}
    for outcome in outcomes:
        grouped.setdefault(outcome["window_id"], []).append(outcome)
    lines = ["", "## Measured intersection-supervisor outcomes", "",
             "These results replay the recorded frames against **all forklifts** using the online intersection supervisor and the viewer's motion policy. They measure the intersection layer only; image-based OOD and TTC layers are excluded because the dataset image mount is not available to this batch evaluator.", "",
             "| Window | Entry-control pass | Clearance pass | Commands observed |", "|---|---:|---:|---|"]
    for window_id, values in grouped.items():
        entry = [v for v in values if v["expected_coverage"] == "entry_control"]
        clearance = [v for v in values if v["expected_coverage"] == "clearance"]
        commands = Counter()
        for value in values:
            commands.update(value["action_counts"])
        lines.append("| {} | {}/{} | {}/{} | {} |".format(
            window_id, sum(v["pass"] for v in entry), len(entry),
            sum(v["pass"] for v in clearance), len(clearance),
            ", ".join("{}={}".format(k, commands[k]) for k in sorted(commands)),
        ))
    total_pass = sum(item["pass"] for item in outcomes)
    lines.extend(["", "Overall measured pass: **{}/{}**. A passing entry-control case receives YIELD at the intersection boundary; a passing clearance case receives CLEAR_INTERSECTION while the AMR is inside.".format(total_pass, len(outcomes)), ""])
    with SUMMARY.open("a") as summary:
        summary.write("\n".join(lines))
    print("Measured {}/{} passing intersection-layer cases".format(total_pass, len(outcomes)))


if __name__ == "__main__":
    main()
