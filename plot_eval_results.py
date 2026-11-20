#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CSV = "./scenario_data/scenario_summary.csv"

OUTDIR = Path("whatif")
OUTDIR.mkdir(exist_ok=True)


df = pd.read_csv(
    CSV,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# Normalize exported fields
# --------------------------------------------------

df["expected_intervention"] = (
    df["expected_action"]
    .astype(str)
    .eq("INTERVENE")
)

df["actual_intervention"] = (
    df["actual_action"]
    .isin(
        [
            "YIELD",
            "AVOID_LEFT",
            "WAIT_BEFORE_INTERSECTION",
            "DEGRADED_MODE",
        ]
    )
)

for col in [
    "trigger_ttc",
    "trigger_trust",
    "max_risk",
    "route_prob",
    "route_uncertainty",
    "maneuver_confidence",
    "topology_confidence",
    "timing_margin_frames",
    "required_ttc",
    "required_risk",
    "baseline_min_ttc",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# --------------------------------------------------
# Generic evaluator
# --------------------------------------------------

def classification_stats(mask):

    predicted = mask.fillna(False)

    expected = df["expected_intervention"]

    tp = (predicted & expected).sum()
    fp = (predicted & ~expected).sum()
    fn = (~predicted & expected).sum()
    tn = (~predicted & ~expected).sum()

    total = tp + fp + fn + tn

    return {
        "pass_rate":
            (tp + tn) / total,

        "false_positive_rate":
            fp / total,

        "miss_rate":
            fn / total,

        "intervention_rate":
            predicted.mean(),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# --------------------------------------------------
# Sweep helper
# --------------------------------------------------

def sweep(
    column,
    thresholds,
    comparator,
):

    result = []

    if column not in df.columns:
        return pd.DataFrame()

    values = df[column]

    for threshold in thresholds:

        mask = comparator(
            values,
            threshold
        )

        row = classification_stats(mask)

        row["threshold"] = threshold

        result.append(row)

    return pd.DataFrame(result)


# --------------------------------------------------
# Parameter sweeps
# --------------------------------------------------

sweeps = {
    "trigger_ttc":
        (
            np.arange(
                0.5,
                5.1,
                0.25
            ),
            lambda x, t: x <= t
        ),

    "trigger_trust":
        (
            np.arange(
                0.30,
                1.01,
                0.05
            ),
            lambda x, t: x <= t
        ),

    "max_risk":
        (
            np.arange(
                0.05,
                1.01,
                0.05
            ),
            lambda x, t: x >= t
        ),

    "route_prob":
        (
            np.arange(
                0.30,
                1.01,
                0.05
            ),
            lambda x, t: x >= t
        ),

    "route_uncertainty":
        (
            np.arange(
                0.05,
                1.01,
                0.05
            ),
            lambda x, t: x <= t
        ),

    "maneuver_confidence":
        (
            np.arange(
                0.30,
                1.01,
                0.05
            ),
            lambda x, t: x >= t
        ),

    "topology_confidence":
        (
            np.arange(
                0.30,
                1.01,
                0.05
            ),
            lambda x, t: x >= t
        ),
}


# --------------------------------------------------
# Run sweeps
# --------------------------------------------------

for parameter, config in sweeps.items():

    thresholds, comparator = config

    frame = sweep(
        parameter,
        thresholds,
        comparator
    )

    if frame.empty:
        print(
            f"Skipping {parameter} "
            "(not exported)"
        )
        continue

    frame.to_csv(
        OUTDIR /
        f"{parameter}_sweep.csv",
        index=False
    )

    plt.figure(figsize=(8, 4))

    plt.plot(
        frame["threshold"],
        frame["pass_rate"],
        label="Pass rate",
    )

    plt.plot(
        frame["threshold"],
        frame["miss_rate"],
        label="Miss rate",
    )

    plt.plot(
        frame["threshold"],
        frame["false_positive_rate"],
        label="False positive",
    )

    plt.grid(True)

    plt.legend()

    plt.title(
        f"What-if Sweep: {parameter}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTDIR /
        f"{parameter}_sweep.png"
    )

    plt.close()


# --------------------------------------------------
# Scenario-specific analysis
# --------------------------------------------------

for scenario in sorted(
    df["required_scenario"]
    .dropna()
    .unique()
):

    subset = df[
        df["required_scenario"]
        == scenario
    ]

    summary = {

        "samples":
            len(subset),

        "pass_rate":
            (
                subset["result"]
                == "PASS"
            ).mean(),

        "late_rate":
            (
                subset["outcome"]
                == "LATE_ACTION"
            ).mean(),

        "miss_rate":
            (
                subset["outcome"]
                == "MISSED_ACTION"
            ).mean(),

        "false_positive_rate":
            (
                subset["outcome"]
                == "FALSE_POSITIVE"
            ).mean(),

        "avg_required_ttc":
            subset["required_ttc"].mean(),

        "avg_route_prob":
            subset["route_prob"].mean(),

        "avg_route_uncertainty":
            subset["route_uncertainty"].mean(),

        "avg_maneuver_conf":
            subset["maneuver_confidence"].mean(),

        "avg_topology_conf":
            subset["topology_confidence"].mean(),
    }

    pd.DataFrame([summary]).to_csv(
        OUTDIR /
        f"{scenario}_summary.csv",
        index=False,
    )


# --------------------------------------------------
# Timing analysis
# --------------------------------------------------

timing = df[
    df["timing_margin_frames"]
    .notna()
]

if len(timing):

    timing[
        "timing_margin_frames"
    ].hist(
        bins=40
    )

    plt.title(
        "Trigger timing margin"
    )

    plt.xlabel(
        "Frames (positive = early)"
    )

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()

    plt.savefig(
        OUTDIR /
        "timing_margin_histogram.png"
    )

    plt.close()


print(
    "What-if analysis written to",
    OUTDIR,
)