#!/usr/bin/env python3
"""Create behavior statistics for cached Scenario 2 replay instances."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCES = ROOT / "JSON-data/scenario_2_instances.json"
CANDIDATES = ROOT / "JSON-data/scenario_2_candidate_windows.json"
REPORT = ROOT / "report_outputs/scenario_2_behavior_summary.md"
RESULTS = ROOT / "report_outputs/scenario_2_behavior_statistics.json"


def result_for(instance):
    actions = instance.get("precomputed_behavior", {}).get("actions", [])
    counts = Counter(actions)
    kind = instance["coverage"]
    protective = counts["STOP"] + counts["YIELD"] + counts["SLOW"]
    if kind in {"collision_avoidance", "head_on_or_crossing"}:
        passed = protective > 0
        criterion = "at least one STOP, YIELD, or SLOW command"
    elif kind == "loading_obstacle":
        passed = protective > 0
        criterion = "at least one protective command while loading context is replayed"
    elif kind == "stationary_clearance":
        passed = counts["GO"] + counts["SLOW"] > 0
        criterion = "GO or SLOW is available (no collision-specific stop required)"
    else:
        passed = counts["GO"] > 0
        criterion = "GO is available during the safe-proceed variant"
    return {"id": instance["id"], "window_id": instance["window_id"], "coverage": kind,
            "action_counts": dict(counts), "pass": passed, "criterion": criterion,
            "min_path_distance_px": instance["desired_behavior"]["minimum_recorded_path_distance_px"]}


def main():
    payload = json.load(INSTANCES.open())
    candidates = {item["id"]: item for item in json.load(CANDIDATES.open())["windows"]}
    outcomes = [result_for(instance) for instance in payload["instances"]]
    json.dump({"method": "Precomputed offline TTC + intersection-supervisor actions", "outcomes": outcomes}, RESULTS.open("w"), indent=2)
    by_window = defaultdict(list)
    by_behavior = defaultdict(list)
    for outcome in outcomes:
        by_window[outcome["window_id"]].append(outcome)
        by_behavior[outcome["coverage"]].append(outcome)
    lines = [
        "# Scenario 2 — measured behavior summary",
        "",
        "Results are calculated from the precomputed offline replay cache. The cache runs the TTC layer for the focal forklift and the intersection layer against all recorded forklifts. This is the same cached behavior used by default in the viewer.",
        "",
        "## Recorded windows and intended robot coverage",
        "",
        "| Recorded window | Forklift behaviour | Scene context | Robot variants and intended coverage |",
        "|---|---|---|---|",
    ]
    for window_id, values in sorted(by_window.items()):
        candidate = candidates[window_id]
        behavior = "Track {}; {}; {:.1f} px/s; route {}→{}; {}".format(
            candidate["focus_track_id"], candidate["maneuver"], candidate["speed_px_s"],
            candidate["route"]["from"], candidate["route"]["to"], candidate["speed_band"],
        )
        context = "{} px from intersection; {} nearby forklifts".format(
            candidate["distance_to_intersection_px"], candidate["nearby_forklift_count"]
        )
        coverage = Counter(value["coverage"] for value in values)
        coverage_text = "; ".join("{} {}".format(count, kind.replace("_", " "))
                                  for kind, count in sorted(coverage.items()))
        lines.append("| {} | {} | {} | {} |".format(window_id, behavior, context, coverage_text))
    lines.extend([
        "",
        "## Overall behavior coverage",
        "",
        "| Intended behavior | Cases | Passes | Pass criterion |",
        "|---|---:|---:|---|",
    ])
    for kind, values in sorted(by_behavior.items()):
        lines.append("| {} | {} | {}/{} | {} |".format(
            kind.replace("_", " "), len(values), sum(v["pass"] for v in values), len(values), values[0]["criterion"]
        ))
    lines.extend(["", "## Per recorded time window", "", "| Window | Passes | Commands observed |", "|---|---:|---|"])
    for window_id, values in sorted(by_window.items()):
        commands = Counter()
        for value in values:
            commands.update(value["action_counts"])
        lines.append("| {} | {}/{} | {} |".format(
            window_id, sum(v["pass"] for v in values), len(values),
            ", ".join("{}={}".format(action, commands[action]) for action in sorted(commands)),
        ))
    total = sum(v["pass"] for v in outcomes)
    lines.extend([
        "", "## Interpretation", "",
        "Overall: **{}/{}** variants meet their stated cache-level pass criterion.".format(total, len(outcomes)),
        "",
        "These are behavior-observation statistics, not a certified collision-safety metric. The current Scenario 2 implementation has generic TTC protection; it does not yet implement separate dedicated policies for notifying a loading-forklift operator, selecting an alternative route, or parking beside a head-on forklift. Those functions should be added before treating the corresponding rows as requirements-compliance results.",
    ])
    REPORT.write_text("\n".join(lines) + "\n")
    print("Wrote {}: {}/{} pass".format(REPORT.relative_to(ROOT), total, len(outcomes)))


if __name__ == "__main__":
    main()
