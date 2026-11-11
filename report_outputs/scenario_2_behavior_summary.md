# Scenario 2 — measured behavior summary

Results are calculated from the precomputed offline replay cache. The cache runs the TTC layer for the focal forklift and the intersection layer against all recorded forklifts. This is the same cached behavior used by default in the viewer.

## Recorded windows and intended robot coverage

| Recorded window | Forklift behaviour | Scene context | Robot variants and intended coverage |
|---|---|---|---|
| scenario_2_window_01 | Track 31; moving; 29.8 px/s; route 1→2; moving | 260.2 px from intersection; 2 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |
| scenario_2_window_02 | Track 45; stationary; 9.7 px/s; route 3→0; slow | 714.2 px from intersection; 1 nearby forklifts | 10 stationary clearance |
| scenario_2_window_03 | Track 54; loading; 0.9 px/s; route 1→0; stationary/very slow | 463.5 px from intersection; 0 nearby forklifts | 10 loading obstacle |
| scenario_2_window_04 | Track 92; moving; 34.2 px/s; route 0→1; moving | 601.3 px from intersection; 2 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |
| scenario_2_window_05 | Track 0; moving; 161.8 px/s; route 0→0; moving | 629.1 px from intersection; 1 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |
| scenario_2_window_06 | Track 1; moving; 27.8 px/s; route 2→0; moving | 438.3 px from intersection; 1 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |
| scenario_2_window_07 | Track 39; moving; 71.5 px/s; route 3→3; moving | 342.2 px from intersection; 1 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |
| scenario_2_window_08 | Track 112; moving; 64.1 px/s; route 0→2; moving | 664.9 px from intersection; 1 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |
| scenario_2_window_09 | Track 16; moving; 116.6 px/s; route 3→1; moving | 232.7 px from intersection; 0 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |
| scenario_2_window_10 | Track 32; moving; 50.9 px/s; route 2→3; moving | 222.9 px from intersection; 0 nearby forklifts | 4 collision avoidance; 3 head on or crossing; 3 safe proceed |

## Overall behavior coverage

| Intended behavior | Cases | Passes | Pass criterion |
|---|---:|---:|---|
| collision avoidance | 32 | 31/32 | at least one STOP, YIELD, or SLOW command |
| head on or crossing | 24 | 24/24 | at least one STOP, YIELD, or SLOW command |
| loading obstacle | 10 | 10/10 | at least one protective command while loading context is replayed |
| safe proceed | 24 | 17/24 | GO is available during the safe-proceed variant |
| stationary clearance | 10 | 10/10 | GO or SLOW is available (no collision-specific stop required) |

## Per recorded time window

| Window | Passes | Commands observed |
|---|---:|---|
| scenario_2_window_01 | 8/10 | GO=8, SLOW=415, STOP=350, YIELD=237 |
| scenario_2_window_02 | 10/10 | GO=25, SLOW=270, STOP=519, YIELD=196 |
| scenario_2_window_03 | 10/10 | GO=18, SLOW=339, STOP=450, YIELD=203 |
| scenario_2_window_04 | 7/10 | GO=18, SLOW=477, STOP=330, YIELD=185 |
| scenario_2_window_05 | 10/10 | GO=35, SLOW=687, STOP=61, YIELD=227 |
| scenario_2_window_06 | 10/10 | GO=573, SLOW=160, STOP=132, YIELD=145 |
| scenario_2_window_07 | 8/10 | GO=452, SLOW=209, STOP=230, YIELD=119 |
| scenario_2_window_08 | 9/10 | GO=203, SLOW=211, STOP=254, YIELD=342 |
| scenario_2_window_09 | 10/10 | GO=862, SLOW=25, STOP=70, YIELD=53 |
| scenario_2_window_10 | 10/10 | GO=773, SLOW=105, STOP=64, YIELD=68 |

## Interpretation

Overall: **92/100** variants meet their stated cache-level pass criterion.

These are behavior-observation statistics, not a certified collision-safety metric. The current Scenario 2 implementation has generic TTC protection; it does not yet implement separate dedicated policies for notifying a loading-forklift operator, selecting an alternative route, or parking beside a head-on forklift. Those functions should be added before treating the corresponding rows as requirements-compliance results.
