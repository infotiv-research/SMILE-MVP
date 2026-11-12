# Scenario replay integration

This integration contains the data definitions and scripts needed to identify
recorded forklift windows for Scenario 1 (intersection operation) and Scenario
2 (operation outside intersections), attach approved AMR routes, evaluate them
with the existing MVP supervisor, and replay them in a separate scenario UI.

## What this adds

- Dataset-backed Scenario 1 and Scenario 2 candidate-window extraction.
- Scenario instances: ten AMR route variants for each selected recorded
  forklift window.
- A headless precompute harness that runs the existing MVP supervisor; it does
  not add or modify any robot safety, TTC, OOD, or decision logic.
- `--scenario` replay mode, including live evaluation by default and optional
  cache-only playback.

The default `python tuve_mvp_2026_04_16.py` viewer remains unchanged.

## Repository artifacts

- `scenario_data/intersection_area.json` — user-defined intersection polygon.
- `scenario_data/robot_operating_area.json` — user-defined AMR operating area.
- `scenario_data/robot_reference_paths.json` — Scenario 1 reference routes.
- `scenario_data/robot_reference_paths_scenario_2.json` — Scenario 2 reference routes.
- `scenario_data/scenario_1_candidate_windows.json` — selected Scenario 1 windows.
- `scenario_data/scenario_2_candidate_windows.json` — selected Scenario 2 windows.
- `scenario_output/` — human-readable window inventories created by the scripts.

## Factory geometry: operating area and intersection

Two user-defined polygons make the scenario definitions spatially explicit:

- `intersection_area.json` defines the Scenario 1 intersection. Its image
  polygon is used to identify recorded forklift entries and to verify that
  Scenario 1 AMR routes cross the intersection.
- `robot_operating_area.json` defines the permitted AMR area. Scenario 1 route
  selection rejects reference paths outside this polygon.

Each file stores the source BEV image, image dimensions, the polygon in BEV
pixels (`polygon_image_px`), and its ground-plane counterpart in metres
(`polygon_ground_world_m`, with `z=0`). Keep both representations consistent
when redefining an area. The scenario viewer overlays both polygons: green for
the operating area and cyan for the intersection.

Reference routes are stored separately in `robot_reference_paths.json` for
Scenario 1 and `robot_reference_paths_scenario_2.json` for Scenario 2. They
are user-approved AMR geometry; forklifts remain entirely recorded data.

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

## Purpose, provenance, and reproducibility

The scenario workflow deliberately separates recorded factory behavior from
simulated AMR behavior:

- Forklift detections, polygons, timestamps, and motion come from the recorded
  dataset. Forklifts are never simulated.
- The trajectory/maneuver prediction file annotates each recorded forklift
  state with route, maneuver, confidence, and uncertainty context.
- Only the AMR path, timing, and resulting supervisor response are simulated.

### Why window extraction uses predictions

The physical candidate event is found from recorded forklift geometry. For
example, Scenario 1 detects a recorded outside-to-inside transition through
the intersection; Scenario 2 detects recorded forklift activity outside it.
Predictions are then used to:

- select a diverse set of route, maneuver, uncertainty, and traffic contexts;
- describe the recorded event in the candidate reports; and
- classify which Scenario 2 AMR response is intended (moving, stationary, or
  loading context).

Consequently, updating the trajectory/maneuver model or its prediction JSON
can change the selected candidate ranking, labels, and intended-coverage
metadata even when the underlying recorded forklift tracks are unchanged.

### Why instance generation uses predictions and the supervisor

Instance generators attach user-defined AMR reference paths to fixed recorded
windows. They use prediction labels to document the intended test behavior and
preserve the recorded prediction data in each replay frame. The supervisor is
**not** used to create forklift behavior or choose a synthetic forklift path.
It is run later during online replay or offline behavior precomputation to
measure how the AMR should respond to the recorded scene.

For a fixed assessment baseline, commit the candidate-window JSON and selected
compressed instance artifact together. Regenerating from a different dataset
or prediction-file version is useful for refreshing the scenario catalogue,
but is not expected to be byte-identical.

## Generate robot-path instances

The generation scripts select and time reference-path variants against the
candidate windows. They use local raw recordings from `JSON-data/`, but write
their generated JSON and reports to the scenario folders:

```bash
python generate_scenario_1_instances.py
python generate_scenario_2_instances.py
python precompute_scenario_behaviors.py
```

Generated uncompressed replay files are written to:

- `scenario_data/scenario_1_instances.json`
- `scenario_data/scenario_2_instances.json`

The committed `.json.gz` variants contain the selected replay artifacts for
sharing. The reports describing intended instance coverage are in:

- `scenario_output/scenario_1_instance_classification.md`
- `scenario_output/scenario_2_instance_classification.md`

### Precompute replay behaviour

`precompute_scenario_behaviors.py` is a headless harness for the existing
`tuve_mvp_2026_04_16.py` implementation. It installs each extracted replay as
the MVP viewer's current data/path and calls its existing prediction lookup,
scene-risk evaluation, and robot-motion methods. It does **not** implement or
modify supervisor policy.

The script writes the compressed replay artifacts consumed by `--scenario`.
All variants of a selected window are first normalised to the same recorded
frame sequence. A completed path is held at its endpoint for remaining replay
frames and marked `GOAL_REACHED` for display.

## Visualize a scenario replay

The regular MVP viewer is unchanged. Use the explicit `--scenario` flag to
launch the separate, read-only scenario replay interface:

```bash
python tuve_mvp_2026_04_16.py --scenario
```

This is the default live mode: each AMR is stepped through the existing MVP
supervisor during playback. For faster cache-only playback, use:

```bash
python tuve_mvp_2026_04_16.py --scenario --precomputed
```

This interface reads the compressed instance artifacts in `scenario_data/` and
provides separate controls for Scenario 1/2, recorded time window, and either
one robot instance or all robot instances in parallel. It does not alter the
normal MVP viewer layout or normal viewer behavior.
