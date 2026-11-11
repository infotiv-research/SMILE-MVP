"""Scenario 1 intersection supervisor used by generation and MVP replay."""

import math

from shapely.geometry import Point, Polygon


def _entry_time(point, velocity_px_s, intersection, horizon_s=3.0, step_s=0.05):
    """Return first predicted intersection entry time, or infinity."""
    if intersection.covers(point):
        return 0.0
    vx, vy = velocity_px_s
    steps = int(horizon_s / step_s)
    for step in range(1, steps + 1):
        time_s = step * step_s
        predicted = Point(point.x + vx * time_s, point.y + vy * time_s)
        if intersection.covers(predicted):
            return time_s
    return math.inf


def evaluate_scenario_1(robot_xy, forklifts, intersection_points, ood_flag=False,
                        uncertainty=0.0, approach_distance_px=160.0,
                        entry_horizon_s=1.0, robot_path_intersects_intersection=True,
                        robot_entry_time_s=math.inf, robot_entry_window_s=3.0):
    """Apply Scenario 1 entry/clearance and fail-safe rules.

    ``forklifts`` contains observed object dictionaries with ``polygon``,
    ``velocity_px_s`` and optional ``maneuver`` fields. The returned action is
    one of GO, YIELD, STOP, or CLEAR_INTERSECTION.
    """
    intersection = Polygon(intersection_points)
    robot = Point(robot_xy)
    robot_inside = intersection.covers(robot)
    threats = []
    obstruction = False

    for forklift in forklifts:
        polygon = Polygon(forklift["polygon"])
        center = polygon.centroid
        maneuver = forklift.get("maneuver", "moving")
        velocity = forklift.get("velocity_px_s", [0.0, 0.0])
        entry_time_s = _entry_time(center, velocity, intersection)
        distance_px = polygon.distance(intersection)
        is_loading_obstruction = maneuver == "loading" and polygon.intersects(intersection)
        if is_loading_obstruction:
            obstruction = True
        approaching = (
            entry_time_s <= entry_horizon_s
            or (distance_px <= approach_distance_px and entry_time_s <= 3.0)
        )
        if approaching:
            threats.append({
                "track_id": forklift.get("track_id"),
                "entry_time_s": entry_time_s,
                "distance_px": distance_px,
            })

    if ood_flag or uncertainty >= 0.7:
        return {"action": "STOP", "reason": "perception_fail_safe", "threats": threats}
    if obstruction:
        return {"action": "STOP", "reason": "loading_obstruction", "threats": threats}
    if threats and robot_inside:
        return {"action": "CLEAR_INTERSECTION", "reason": "approaching_forklift", "threats": threats}
    if threats and not robot_path_intersects_intersection:
        return {"action": "GO", "reason": "non_conflicting_robot_path", "threats": threats}
    if threats and robot_entry_time_s > robot_entry_window_s:
        return {"action": "GO", "reason": "robot_not_yet_approaching_entry", "threats": threats}
    if threats:
        return {"action": "YIELD", "reason": "entry_control", "threats": threats}
    return {"action": "GO", "reason": "no_conflicting_entry", "threats": threats}
