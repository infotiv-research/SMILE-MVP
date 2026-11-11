# Scenario 1 — recorded-data replay set

All forklift detections, BEV polygons, timestamps, route and manoeuvre classifications, and OOD inputs are replayed from the recorded dataset. Only the AMR path is simulated. Every generated AMR trajectory is a suffix of a user-approved route in `JSON-data/robot_reference_paths.json`, constrained to `JSON-data/robot_operating_area.json`, and crosses the valid overlap with the intersection.

## Batch

- Recorded forklift windows: 10
- AMR route variants per window: 10
- Replay instances: 100
- Window: 40 frames before the recorded entry through at least 15 frames after the recorded forklift clears the intersection
- AMR route context: at least 1 second before entering and 1 second after leaving the intersection
- AMR speed: preserved from the selected user reference path
- Coverage: clearance=30, entry_control=70

This batch contains moving-forklift entry-control examples with AMR through-routes across the intersection. The current recordings contain no loading or stationary detections within the selected intersection, so those claims are intentionally excluded. Export the MVP evidence logs before calculating recall, false positives, or any Vinnova-facing performance metric.
