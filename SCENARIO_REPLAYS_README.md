# Scenario replay handoff

The committed replay artifacts are gzip-compressed to keep the repository practical to clone. The viewer reads `.json.gz` files directly.

## View Scenario 1

```bash
conda activate mot
python tuve_mvp_2026_04_16.py \
  --scenario-replay JSON-data/scenario_1_instances.json.gz \
  --reference-paths JSON-data/robot_reference_paths.json
```

## View Scenario 2

```bash
conda activate mot
python tuve_mvp_2026_04_16.py \
  --scenario-replay JSON-data/scenario_2_instances.json.gz \
  --reference-paths JSON-data/robot_reference_paths_scenario_2.json
```

Use the recorded-window and robot-path controls to select one replay, or select all paths. Precomputed replay behavior is enabled by default.

## Regenerate artifacts

The confidential recording must be available through the paths in `mvp-config.json` to regenerate these outputs.

```bash
python identify_scenario_1_windows.py
python generate_scenario_1_instances.py
python evaluate_scenario_1_replays.py

python identify_scenario_2_windows.py
python generate_scenario_2_instances.py
python summarize_scenario_2_behaviors.py

python precompute_replay_behaviors.py
gzip -c JSON-data/scenario_1_instances.json > JSON-data/scenario_1_instances.json.gz
gzip -c JSON-data/scenario_2_instances.json > JSON-data/scenario_2_instances.json.gz
```

Do not commit confidential raw recordings such as `JSON-data/final.json`, generated uncompressed replay JSON, dataset mounts, or OOD image folders.
