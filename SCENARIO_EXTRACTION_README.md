# Scenario window extraction

This integration contains the data definitions and scripts needed to identify
recorded forklift windows for Scenario 1 (intersection operation) and Scenario
2 (operation outside intersections).

## Repository artifacts

- `scenario_data/intersection_area.json` — user-defined intersection polygon.
- `scenario_data/robot_operating_area.json` — user-defined AMR operating area.
- `scenario_data/robot_reference_paths.json` — Scenario 1 reference routes.
- `scenario_data/robot_reference_paths_scenario_2.json` — Scenario 2 reference routes.
- `scenario_data/scenario_1_candidate_windows.json` — selected Scenario 1 windows.
- `scenario_data/scenario_2_candidate_windows.json` — selected Scenario 2 windows.
- `scenario_output/` — human-readable window inventories created by the scripts.

## Prerequisites

The raw confidential recording is intentionally not stored in Git. Configure
the local dataset and ensure these local files exist:

- `JSON-data/final.json`
- `JSON-data/predictions_2026_02_27.json`

Activate the MOT environment before running the scripts:

```bash
conda activate mot
```

## Recreate the candidate windows

```bash
python identify_scenario_1_windows.py
python identify_scenario_2_windows.py
```

Scenario 1 finds diverse observed forklift entries into the defined
intersection. Scenario 2 finds diverse forklift activity with the focus track
outside the intersection. Both scripts overwrite only the corresponding
candidate JSON and Markdown files in `scenario_data/` and `scenario_output/`.

## Review

Read the generated tables in:

- `scenario_output/scenario_1_candidate_time_windows.md`
- `scenario_output/scenario_2_candidate_time_windows.md`

The candidate JSON includes source frame indices, track IDs, route/maneuver
classification context, and nearby-forklift information for later replay and
robot-path generation.
