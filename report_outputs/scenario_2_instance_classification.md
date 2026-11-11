# Scenario 2 — robot-path behavior coverage

Each recorded outside-intersection window has ten AMR route variants. Every selected route comes within 60 px of the recorded focus forklift trajectory; when fewer than ten directed paths are available, only relevant geometry is reused with different arrival timing.

| Window | Coverage |
|---|---|
| scenario_2_window_01 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |
| scenario_2_window_02 | stationary_clearance, stationary_clearance, stationary_clearance, stationary_clearance, stationary_clearance, stationary_clearance, stationary_clearance, stationary_clearance, stationary_clearance, stationary_clearance |
| scenario_2_window_03 | loading_obstacle, loading_obstacle, loading_obstacle, loading_obstacle, loading_obstacle, loading_obstacle, loading_obstacle, loading_obstacle, loading_obstacle, loading_obstacle |
| scenario_2_window_04 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |
| scenario_2_window_05 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |
| scenario_2_window_06 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |
| scenario_2_window_07 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |
| scenario_2_window_08 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |
| scenario_2_window_09 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |
| scenario_2_window_10 | collision_avoidance, collision_avoidance, collision_avoidance, collision_avoidance, head_on_or_crossing, head_on_or_crossing, head_on_or_crossing, safe_proceed, safe_proceed, safe_proceed |

These variants exercise the existing online TTC/trajectory safety layer. Loading/stationary entries are recorded maneuver contexts; a dedicated Scenario 2 maneuver policy can be added before measuring rule-specific pass/fail results.
