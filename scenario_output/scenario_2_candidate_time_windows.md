# Scenario 2 — candidate recorded time windows

Each window is a 10-second recorded forklift segment whose focus track remains at least 180 px from the defined intersection at the centre frame.

| ID | Source frames | Focus forklift behaviour | Prediction | Local traffic |
|---|---|---|---|---|
| scenario_2_window_01 | 9243–9343 | track 31; 29.8 px/s (moving); 260.2 px from intersection | moving; route 1→2; uncertainty 0.2731262567688352 | 2 nearby |
| scenario_2_window_02 | 10159–10259 | track 45; 9.7 px/s (slow); 714.2 px from intersection | stationary; route 3→0; uncertainty 0.02308797489756842 | 1 nearby |
| scenario_2_window_03 | 15532–15632 | track 54; 0.9 px/s (stationary/very slow); 463.5 px from intersection | loading; route 1→0; uncertainty 0.20788380322311162 | 0 nearby |
| scenario_2_window_04 | 25402–25502 | track 92; 34.2 px/s (moving); 601.3 px from intersection | moving; route 0→1; uncertainty 0.20801817201967843 | 2 nearby |
| scenario_2_window_05 | 70–170 | track 0; 161.8 px/s (moving); 629.1 px from intersection | moving; route 0→0; uncertainty 0.4622079468243001 | 1 nearby |
| scenario_2_window_06 | 4217–4317 | track 1; 27.8 px/s (moving); 438.3 px from intersection | moving; route 2→0; uncertainty 0.5474668735391301 | 1 nearby |
| scenario_2_window_07 | 8818–8918 | track 39; 71.5 px/s (moving); 342.2 px from intersection | moving; route 3→3; uncertainty 0.6001882781219229 | 1 nearby |
| scenario_2_window_08 | 29795–29895 | track 112; 64.1 px/s (moving); 664.9 px from intersection | moving; route 0→2; uncertainty 0.6405576265681217 | 1 nearby |
| scenario_2_window_09 | 5555–5655 | track 16; 116.6 px/s (moving); 232.7 px from intersection | moving; route 3→1; uncertainty 0.0818660941092304 | 0 nearby |
| scenario_2_window_10 | 7666–7766 | track 32; 50.9 px/s (moving); 222.9 px from intersection | moving; route 2→3; uncertainty 0.006280929732737639 | 0 nearby |

These are recorded scene candidates only. The next step is to assign AMR routes that test TTC monitoring, head-on/crossing avoidance, loading/stationary handling, and fail-safe responses.
