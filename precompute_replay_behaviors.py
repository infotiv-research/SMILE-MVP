#!/usr/bin/env python3
"""Cache per-frame AMR positions/actions so replay playback is visualization-only."""

import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from scenario_supervisor import evaluate_scenario_1


ROOT = Path(__file__).resolve().parent
OPERATING_AREA = Polygon(json.load((ROOT / "JSON-data/robot_operating_area.json").open())["polygon_image_px"])


def move(state, intersection):
    if state["action"] == "STOP" or state["done"]:
        return
    if state["action"] == "PARK_SAFE":
        if state["parking_arrived"]:
            return
        tx, ty = state["parking_target"]
        dx, dy = tx - state["xy"][0], ty - state["xy"][1]
        distance = math.hypot(dx, dy)
        if distance <= 3:
            state["xy"] = [tx, ty]; state["parking_arrived"] = True
        else:
            step = min(state["speed"], distance)
            state["xy"][0] += dx / distance * step; state["xy"][1] += dy / distance * step
        return
    speed = state["speed"] * (1.5 if state["action"] == "CLEAR_INTERSECTION" else 0.5 if state["action"] == "SLOW" else 1.0)
    if state["action"] == "YIELD":
        distance = Point(state["xy"]).distance(intersection)
        if distance <= 3:
            return
        speed = min(speed, distance - 3)
    while speed > 0 and state["index"] < len(state["path"]):
        target = state["path"][state["index"]]
        dx, dy = target[0] - state["xy"][0], target[1] - state["xy"][1]
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            state["index"] += 1
        elif distance <= speed:
            state["xy"] = list(target); state["index"] += 1; speed -= distance
        else:
            state["xy"][0] += dx / distance * speed; state["xy"][1] += dy / distance * speed; speed = 0
    state["done"] = state["index"] >= len(state["path"])


def simulate(instance, intersection, allow_safe_parking=False):
    state = {"path": [list(p) for p in instance["robot_path"]], "xy": list(instance["robot_path"][0]),
             "speed": float(instance["robot_speed_px_per_frame"]), "index": 1, "action": "GO", "done": False,
             "parking_target": None, "parking_arrived": False, "parking_clear_frames": 0}
    positions, actions = [], []
    for frame in instance["frames"]:
        move(state, intersection)
        remaining = [state["xy"]] + state["path"][state["index"]:]
        enters = len(remaining) >= 2 and intersection.intersects(LineString(remaining))
        entry_time = Point(state["xy"]).distance(intersection) / (state["speed"] / 0.1) if enters else math.inf
        predictions = {int(p["track_id"]): p for p in frame.get("predictions", [])}
        forklifts = [{"track_id": o["track_id"], "polygon": o["polygon"],
                      "velocity_px_s": o.get("velocity_px_s", [0, 0]),
                      "maneuver": predictions.get(int(o["track_id"]), {}).get("mode", "moving")}
                     for o in frame["objects"]]
        result = evaluate_scenario_1(state["xy"], forklifts, list(intersection.exterior.coords)[:-1],
                                     robot_path_intersects_intersection=enters, robot_entry_time_s=entry_time)
        # Mirror the viewer's scene-level TTC layer: the shortest closing
        # time of any forklift controls the generic protective action.
        ttc_action = "GO"
        if state["index"] < len(state["path"]):
            target = state["path"][state["index"]]
            hx, hy = target[0] - state["xy"][0], target[1] - state["xy"][1]
            norm = math.hypot(hx, hy)
            if norm > 1e-9:
                minimum_ttc = math.inf
                for obj in frame["objects"]:
                    polygon = obj["polygon"]
                    cx = sum(p[0] for p in polygon) / len(polygon)
                    cy = sum(p[1] for p in polygon) / len(polygon)
                    forward = ((cx - state["xy"][0]) * hx + (cy - state["xy"][1]) * hy) / norm
                    if forward > 0:
                        robot_speed = state["speed"] / 0.1
                        heading_x, heading_y = hx / norm, hy / norm
                        object_velocity = obj.get("velocity_world", [0.0, 0.0])
                        closing = robot_speed - (
                            float(object_velocity[0]) * heading_x
                            + float(object_velocity[1]) * heading_y
                        )
                        if closing > 0:
                            minimum_ttc = min(minimum_ttc, forward / closing)
                if minimum_ttc < 2: ttc_action = "STOP"
                elif minimum_ttc < 4: ttc_action = "YIELD"
                elif minimum_ttc < 8: ttc_action = "SLOW"
        priority = {"GO": 0, "SLOW": 1, "YIELD": 2, "CLEAR_INTERSECTION": 2, "STOP": 3}
        state["action"] = result["action"] if priority[result["action"]] >= priority[ttc_action] else ttc_action
        # Same head-on safe-parking condition as the online viewer.
        head_on = False
        if state["index"] < len(state["path"]):
            target = state["path"][state["index"]]
            hx, hy = target[0] - state["xy"][0], target[1] - state["xy"][1]
            norm = math.hypot(hx, hy)
            if norm > 1e-9:
                hx, hy = hx / norm, hy / norm
                for obj in frame["objects"]:
                    polygon = obj["polygon"]
                    cx = sum(p[0] for p in polygon) / len(polygon)
                    cy = sum(p[1] for p in polygon) / len(polygon)
                    forward = (cx - state["xy"][0]) * hx + (cy - state["xy"][1]) * hy
                    vx, vy = obj.get("velocity_px_s", [0.0, 0.0])
                    velocity_norm = math.hypot(vx, vy)
                    if 0 < forward <= 120 and velocity_norm >= 5 and (vx / velocity_norm) * hx + (vy / velocity_norm) * hy <= -0.5:
                        head_on = True; break
        if allow_safe_parking and head_on:
            if state["parking_target"] is None:
                _, border = nearest_points(Point(state["xy"]), OPERATING_AREA.boundary)
                state["parking_target"] = [border.x, border.y]
            state["parking_clear_frames"] = 0
            state["action"] = "PARK_SAFE"
        elif allow_safe_parking and state["parking_target"] is not None:
            state["parking_clear_frames"] += 1
            if state["parking_arrived"] and state["parking_clear_frames"] >= 10:
                state["index"] = min(range(len(state["path"])), key=lambda i: math.dist(state["path"][i], state["xy"]))
                state["index"] = max(1, state["index"])
                state["parking_target"] = None; state["parking_arrived"] = False; state["parking_clear_frames"] = 0
                state["action"] = "GO"
            else:
                state["action"] = "PARK_SAFE"
        positions.append([round(state["xy"][0], 3), round(state["xy"][1], 3)])
        actions.append(state["action"])
    return {"positions": positions, "actions": actions, "method": "offline scene-level TTC + intersection supervisor; all recorded forklifts"}


def process(path):
    payload = json.load(path.open())
    intersection = Polygon(payload["intersection_polygon_px"])
    for instance in payload["instances"]:
        instance["precomputed_behavior"] = simulate(
            instance, intersection,
            allow_safe_parking=str(payload.get("scenario", "")).startswith("Scenario 2"),
        )
    json.dump(payload, path.open("w"), indent=2)
    print("Cached {} behaviors in {}".format(len(payload["instances"]), path.relative_to(ROOT)))


if __name__ == "__main__":
    process(ROOT / "JSON-data/scenario_1_instances.json")
    process(ROOT / "JSON-data/scenario_2_instances.json")
