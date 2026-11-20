#!/usr/bin/env python3

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_instances(path):

    data = json.load(
        gzip.open(path, "rt")
    )

    rows = []

    for inst in data.get(
        "instances",
        []
    ):

        s = (
            inst
            .get(
                "precomputed_behavior",
                {}
            )
            .get(
                "summary",
                {}
            )
        )

        t = (
            s.get("trigger")
            or {}
        )

        b = (
            s.get("baseline")
            or {}
        )

        rows.append({

            "scenario":
                path.stem,

            "instance_id":
                inst.get("id"),

            "trigger_ttc":
                t.get("ttc"),

            "trigger_trust":
                t.get("trust"),

            "trigger_action":
                t.get("action"),

            "outcome":
                s.get("outcome"),

            "result":
                s.get("result"),

            "expected_action":
                s.get(
                    "expected_action"
                ),

            "baseline_min_ttc":
                b.get(
                    "min_ttc"
                ),

            "baseline_collision":
                b.get(
                    "collision"
                ),

            "safety_critical":
                (
                    s.get(
                        "expected_action"
                    )
                    != "PROCEED"
                ),
        })

    return rows


def evaluate_ttc(
    df,
    threshold
):

    pass_count = 0
    fail_count = 0
    false_positive = 0

    for row in df.itertuples():

        critical = bool(
            row.safety_critical
        )

        ttc = row.trigger_ttc

        action = (
            row.trigger_action
            or "PROCEED"
        )

        if critical:

            passed = (
                pd.notna(ttc)
                and ttc >= threshold
            )

        else:

            passed = (
                action
                == "PROCEED"
            )

            if (
                action
                != "PROCEED"
            ):
                false_positive += 1

        if passed:
            pass_count += 1
        else:
            fail_count += 1

    total = (
        pass_count
        + fail_count
    )

    return {
        "threshold":
            threshold,

        "pass_rate":
            pass_count / total,

        "fail_rate":
            fail_count / total,

        "false_positive_rate":
            false_positive
            / total,
    }


def evaluate_trust(
    df,
    threshold
):

    triggered = (
        df.trigger_trust
        < threshold
    )

    return {
        "threshold":
            threshold,

        "trigger_rate":
            triggered.mean(),
    }


def sweep_ttc(df):

    rows = []

    for th in (
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    ):

        rows.append(
            evaluate_ttc(
                df,
                th
            )
        )

    return pd.DataFrame(
        rows
    )


def sweep_trust(df):

    rows = []

    for th in (
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ):

        rows.append(
            evaluate_trust(
                df,
                th
            )
        )

    return pd.DataFrame(
        rows
    )


def plot_sweep(
    frame,
    x,
    y,
    title,
    path,
):

    plt.figure(
        figsize=(7,4)
    )

    plt.plot(
        frame[x],
        frame[y],
        marker="o"
    )

    plt.grid(True)

    plt.title(title)

    plt.tight_layout()

    plt.savefig(path)

    plt.close()


def main():

    files = (
        sorted(
            Path(
                "./scenario_data"
            ).glob(
                "scenario_*_instances.json.gz"
            )
        )
        or
        sorted(
            Path(".").glob(
                "scenario_*_instances.json.gz"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            "No scenario files"
        )

    rows = []

    for f in files:
        rows.extend(
            load_instances(f)
        )

    df = pd.DataFrame(rows)
    out = (
        files[0].parent
        /
        "scenario_analysis"
    )

    out.mkdir(
        exist_ok=True
    )

    critical = (
        df.safety_critical
        .value_counts()
    )

    critical.to_csv(
        out /
        "safety_critical_summary.csv"
    )    



    ttc = sweep_ttc(df)

    trust = sweep_trust(df)

    ttc.to_csv(
        out /
        "ttc_threshold_sweep.csv",
        index=False
    )

    trust.to_csv(
        out /
        "trust_threshold_sweep.csv",
        index=False
    )

    plot_sweep(
        ttc,
        "threshold",
        "pass_rate",
        "Pass rate vs TTC threshold",
        out /
        "pass_rate_vs_ttc.png",
    )

    plot_sweep(
        ttc,
        "threshold",
        "fail_rate",
        "Fail rate vs TTC threshold",
        out /
        "fail_rate_vs_ttc.png",
    )

    plot_sweep(
        trust,
        "threshold",
        "trigger_rate",
        "Trigger rate vs trust threshold",
        out /
        "trust_threshold_effect.png",
    )
    plot_sweep(
        ttc,
        "threshold",
        "false_positive_rate",
        "False positive rate vs TTC threshold",
        out /
        "false_positive_vs_ttc.png",
    )

    print(
        f"Analysis written to {out}"
    )


if __name__ == "__main__":
    main()