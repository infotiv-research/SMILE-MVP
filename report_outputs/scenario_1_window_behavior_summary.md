# Scenario 1 — selected windows and robot-behaviour coverage

Each row is one recorded forklift scene. Forklift detections, classifications and timing are recorded; only the AMR routes are simulated. All 10 AMR variants per row are evaluated against **all forklifts** in the replay, not only the named focal forklift.

| Recorded window | Forklift behaviour | Scene context | Robot variants and intended coverage |
|---|---|---|---|
| scenario_1_window_01 | Track 16; north→west; 4.8 s inside; route 3→1; moving | 4 distinct / peak 4 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_2 (forward); Path_16 (forward); Path_3 (reverse); Path_5 (forward); Path_6 (reverse); Path_7 (forward); Path_18 (forward); Path_19 (reverse); Path_20 (forward); Path_21 (reverse) |
| scenario_1_window_02 | Track 59; south→south; 24.0 s inside; route 1→1; moving | 3 distinct / peak 2 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_7 (forward); Path_16 (forward); Path_10 (forward); Path_14 (reverse); Path_18 (forward); Path_19 (reverse); Path_20 (forward); Path_21 (reverse); Path_26 (forward); Path_30 (forward) |
| scenario_1_window_03 | Track 48; east→north; 1.2 s inside; route 0→0; moving | 1 distinct / peak 1 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_14 (forward); Path_16 (forward); Path_15 (reverse); Path_17 (reverse); Path_18 (forward); Path_19 (reverse); Path_20 (forward); Path_21 (reverse); Path_26 (forward); Path_30 (forward) |
| scenario_1_window_04 | Track 120; west→south; 4.1 s inside; route 0→1; moving | 3 distinct / peak 3 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_19 (forward); Path_30 (forward); Path_20 (forward); Path_21 (reverse); Path_22 (reverse); Path_23 (forward); Path_26 (forward); Path_31 (reverse); Path_33 (forward); Path_37 (forward) |
| scenario_1_window_05 | Track 73; north→south; 6.1 s inside; route 2→1; moving | 4 distinct / peak 3 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_24 (forward); Path_30 (forward); Path_25 (forward); Path_26 (forward); Path_31 (reverse); Path_32 (forward); Path_33 (forward); Path_34 (reverse); Path_37 (forward); Path_2 (forward) |
| scenario_1_window_06 | Track 30; north→south; 4.8 s inside; route 4→1; moving | 3 distinct / peak 3 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_31 (forward); Path_33 (forward); Path_32 (forward); Path_34 (reverse); Path_37 (forward); Path_38 (forward); Path_40 (forward); Path_2 (forward); Path_5 (forward); Path_7 (forward) |
| scenario_1_window_07 | Track 31; south→north; 6.1 s inside; route 1→2; moving | 2 distinct / peak 2 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_37 (forward); Path_16 (forward); Path_38 (forward); Path_40 (forward); Path_2 (forward); Path_5 (forward); Path_6 (reverse); Path_7 (forward); Path_18 (forward); Path_19 (reverse) |
| scenario_1_window_08 | Track 39; north→north; 1.2 s inside; route 3→2; moving | 2 distinct / peak 2 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_4 (forward); Path_16 (forward); Path_5 (forward); Path_6 (reverse); Path_7 (forward); Path_10 (reverse); Path_18 (forward); Path_19 (reverse); Path_20 (forward); Path_21 (reverse) |
| scenario_1_window_09 | Track 41; south→north; 5.1 s inside; route 1→4; moving | 1 distinct / peak 1 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_9 (forward); Path_16 (forward); Path_10 (forward); Path_14 (reverse); Path_18 (forward); Path_19 (reverse); Path_20 (forward); Path_21 (reverse); Path_26 (forward); Path_30 (forward) |
| scenario_1_window_10 | Track 46; west→north; 3.5 s inside; route 0→2; moving | 1 distinct / peak 1 nearby forklifts | 7 entry-control yield/delay; 3 clearance/occupancy. Routes: Path_16 (forward); Path_17 (reverse); Path_18 (forward); Path_19 (reverse); Path_20 (forward); Path_21 (reverse); Path_26 (forward); Path_30 (forward); Path_31 (reverse); Path_33 (forward) |

## Scope and interpretation

- The 70 entry-control variants time the AMR to approach after the recorded forklift threat, so it should delay entry at the intersection boundary when required.
- The 30 clearance variants time the AMR to already be in the intersection when the recorded forklift approaches, so it should vacate the area promptly.
- Every selected route stays within the robot operating area and crosses the intersection. Forward and reverse directions are treated as different route variants.
- This catalogue describes intended test coverage. Reported decision outcomes should be derived from the MVP evidence log after replaying the cases, especially because all nearby forklifts now participate in the online safety decision.

## Measured intersection-supervisor outcomes

These results replay the recorded frames against **all forklifts** using the online intersection supervisor and the viewer's motion policy. They measure the intersection layer only; image-based OOD and TTC layers are excluded because the dataset image mount is not available to this batch evaluator.

| Window | Entry-control pass | Clearance pass | Commands observed |
|---|---:|---:|---|
| scenario_1_window_01 | 7/7 | 3/3 | CLEAR_INTERSECTION=142, GO=1269, YIELD=360 |
| scenario_1_window_02 | 7/7 | 3/3 | CLEAR_INTERSECTION=180, GO=1424, YIELD=1626 |
| scenario_1_window_03 | 0/7 | 3/3 | CLEAR_INTERSECTION=37, GO=1591, YIELD=69 |
| scenario_1_window_04 | 7/7 | 3/3 | CLEAR_INTERSECTION=13, GO=1160, YIELD=579 |
| scenario_1_window_05 | 7/7 | 3/3 | CLEAR_INTERSECTION=339, GO=660, YIELD=789 |
| scenario_1_window_06 | 7/7 | 0/3 | GO=96, YIELD=1723 |
| scenario_1_window_07 | 7/7 | 3/3 | CLEAR_INTERSECTION=386, GO=1994, YIELD=440 |
| scenario_1_window_08 | 3/7 | 3/3 | CLEAR_INTERSECTION=426, GO=1262, YIELD=103 |
| scenario_1_window_09 | 7/7 | 3/3 | CLEAR_INTERSECTION=140, GO=1276, YIELD=349 |
| scenario_1_window_10 | 7/7 | 3/3 | CLEAR_INTERSECTION=113, GO=1212, YIELD=350 |

Overall measured pass: **86/100**. A passing entry-control case receives YIELD at the intersection boundary; a passing clearance case receives CLEAR_INTERSECTION while the AMR is inside.
