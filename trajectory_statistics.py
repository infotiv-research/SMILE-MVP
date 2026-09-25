#!/usr/bin/env python3
"""Generate trajectory-only statistics and figures from one selected JSON source.

Segmentation images are not read or required. The default source is the native
tracker export ``final.json``. Use --use-object-tracks for the distinct legacy
BEV export. Outputs are always placed in a source-labelled directory.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, PowerNorm
from PIL import Image
from tqdm.auto import tqdm

from extract_report_statistics import load_tracks, plot_track_figures, quantiles, write_csv


matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})


def load_metric_motion(path: Path) -> list[dict]:
    """Read native world-space positions and velocities; no BEV conversion."""
    with path.open() as handle:
        frames = json.load(handle)
    if not isinstance(frames, list):
        raise ValueError(f"Expected frame-based JSON list in {path}")
    rows = []
    for frame_index, frame in enumerate(tqdm(frames, desc="Reading metric motion", unit="frame", dynamic_ncols=True)):
        for obj in frame.get("object_list", []):
            position = obj.get("position_3d", {})
            velocity = obj.get("velocity_3d", {})
            if any(position.get(key) is None for key in ("x", "y")) or any(velocity.get(key) is None for key in ("x", "y")):
                continue
            vx, vy = float(velocity["x"]), float(velocity["y"])
            box = obj.get("bbox_moving_avg", {})
            polygon = np.asarray(obj.get("associated_polygon", []), dtype=float)
            width, length = box.get("width"), box.get("length")
            if (width is None or length is None) and polygon.ndim == 2 and polygon.shape[0] >= 3:
                width = np.ptp(polygon[:, 0]) if width is None else width
                length = np.ptp(polygon[:, 1]) if length is None else length
            heading = obj.get("heading", {}).get("deg")
            rows.append({"frame": frame_index, "track_id": obj.get("track_id", ""),
                         "world_x_m": float(position["x"]), "world_y_m": float(position["y"]),
                         "velocity_x_mps": vx, "velocity_y_mps": vy, "speed_mps": float(np.hypot(vx, vy)),
                         "width_m": None if width is None else float(width) * .05,
                         "length_m": None if length is None else float(length) * .05,
                         "orientation_deg": None if heading is None else float(heading) % 180.0})
    return rows


def metric_motion_from_bev_tracks(tracks: list[dict]) -> list[dict]:
    """Exact grid conversion for the MVP BEV: 5 cm cells, 10 Hz frames."""
    rows = []
    for row in tracks:
        vx = row["velocity_x_px_per_frame"] * .5 if np.isfinite(row["velocity_x_px_per_frame"]) else float("nan")
        vy = -row["velocity_y_px_per_frame"] * .5 if np.isfinite(row["velocity_y_px_per_frame"]) else float("nan")
        rows.append({"frame": row["frame"], "track_id": row["object_id"],
                     "world_x_m": -26.7 + .05 * row["center_x_px"],
                     "world_y_m": 11.4 - .05 * row["center_y_px"],
                     "velocity_x_mps": vx, "velocity_y_mps": vy, "speed_mps": float(np.hypot(vx, vy)),
                     "width_m": .05 * row["width_px"], "length_m": .05 * row["height_px"],
                     "orientation_deg": row["orientation_deg"] if row["orientation_available"] else None})
    return rows


def save_metric_figure(fig: plt.Figure, output: Path) -> None:
    """Save a vector PDF and a report-friendly, fixed-width PNG."""
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    # World maps deliberately become wide to retain their equal metric x/y
    # scale.  Downsample the final raster to a predictable report-friendly
    # width instead of leaving a multi-thousand-pixel PNG on disk.
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    buffer.seek(0)
    with Image.open(buffer) as image:
        output_width = 1920
        output_height = max(1, round(image.height * output_width / image.width))
        image.resize((output_width, output_height), Image.Resampling.LANCZOS).save(output.with_suffix(".png"))
    plt.close(fig)


def metric_map_axes(x: np.ndarray, y: np.ndarray, *, colorbar: bool = True) -> tuple[plt.Figure, plt.Axes]:
    """Create an equal-scale world map whose axes can use the full figure height.

    A fixed, moderately wide canvas makes a 3.6:1 world-space region letterbox
    vertically once ``ax.set_aspect('equal')`` is applied.  The colourbar then
    looks much taller than the actual map.  Size the canvas from the data's
    physical x/y extent so the map, rather than unused horizontal space, fills
    the colourbar height.
    """
    x_span = max(float(np.ptp(x)), 1e-6)
    y_span = max(float(np.ptp(y)), 1e-6)
    xy_aspect = x_span / y_span
    canvas_height_in = 7.0
    # ``tight_layout`` reserves appreciable width for the y label, tick labels,
    # colourbar and its label.  Include that space explicitly; otherwise the
    # equal-aspect axes shrink vertically, while the colourbar remains tall.
    extra_width_in = 5.2 if colorbar else 2.5
    fig, ax = plt.subplots(figsize=(xy_aspect * canvas_height_in + extra_width_in, canvas_height_in))
    return fig, ax


def style_metric_map(ax: plt.Axes, title: str) -> None:
    """Apply typography that remains legible in a 1920-pixel-wide export."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=34)
    ax.set_xlabel("World x [m]", fontsize=28)
    ax.set_ylabel("World y [m]", fontsize=28)
    ax.tick_params(axis="both", labelsize=24)


def plot_centered_oriented_boxes(rows: list[dict], output: Path, *, width_key: str, height_key: str,
                                 unit: str, title: str) -> int:
    """Overlay every valid oriented box after translating its centre to (0, 0)."""
    widths = np.asarray([row.get(width_key, np.nan) for row in rows], dtype=float)
    heights = np.asarray([row.get(height_key, np.nan) for row in rows], dtype=float)
    angles = np.asarray([row.get("orientation_deg", np.nan) for row in rows], dtype=float)
    valid = np.isfinite(widths) & np.isfinite(heights) & np.isfinite(angles) & (widths > 0) & (heights > 0)
    widths, heights, angles = widths[valid], heights[valid], np.deg2rad(angles[valid])
    if not len(widths):
        return 0

    # Four closed line segments per rectangle.  The positions are deliberately
    # omitted: translating all centres to the origin isolates the joint box
    # size-and-orientation distribution.
    local_corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]])
    corners = local_corners[None, :, :] * np.column_stack([widths, heights])[:, None, :]
    cosine, sine = np.cos(angles), np.sin(angles)
    rotation = np.stack([np.stack([cosine, -sine], axis=1), np.stack([sine, cosine], axis=1)], axis=1)
    corners = corners @ np.swapaxes(rotation, 1, 2)
    segments = np.stack([corners[:, :-1], corners[:, 1:]], axis=2).reshape(-1, 2, 2)

    fig, ax = plt.subplots(figsize=(8, 8))
    collection = LineCollection(segments, colors="#2b6cb0", linewidths=.35, alpha=.025, rasterized=True)
    ax.add_collection(collection)
    extent = float(np.max(np.abs(corners))) * 1.08
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="black", lw=.6, alpha=.4)
    ax.axvline(0, color="black", lw=.6, alpha=.4)
    ax.set(title=f"{title} (n={len(widths):,})", xlabel=f"Centred x [{unit}]", ylabel=f"Centred y [{unit}]")
    save_metric_figure(fig, output)
    return int(len(widths))


def plot_metric_motion(rows: list[dict], output: Path, source: str) -> dict:
    x = np.asarray([row["world_x_m"] for row in rows])
    y = np.asarray([row["world_y_m"] for row in rows])
    vx = np.asarray([row["velocity_x_mps"] for row in rows])
    vy = np.asarray([row["velocity_y_mps"] for row in rows])
    speed = np.asarray([row["speed_mps"] for row in rows])
    valid_motion = np.isfinite(speed)
    p95 = float(np.quantile(speed[valid_motion], .95))

    fig, ax = metric_map_axes(x[valid_motion], y[valid_motion])
    points = ax.scatter(x[valid_motion], y[valid_motion], c=speed[valid_motion], s=20, cmap="viridis", alpha=.5, rasterized=True,
                        norm=PowerNorm(gamma=.5, vmin=0, vmax=max(p95, 1e-6)))
    style_metric_map(ax, "World-space position and measured velocity")
    colorbar = fig.colorbar(points, ax=ax, label="Measured speed [m/s; sqrt scale, capped at p95]")
    colorbar.ax.tick_params(labelsize=20)
    colorbar.set_label("Measured speed [m/s; sqrt scale, capped at p95]", fontsize=28)
    save_metric_figure(fig, output / "metric_position_velocity")

    upper = float(np.quantile(speed[valid_motion], .995))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(speed[valid_motion & (speed <= upper)], bins=50, color="#277f8e")
    axes[0].set(title="Measured speed distribution", xlabel="Speed [m/s]", ylabel="Object observations")
    axes[1].hist(vx[valid_motion], bins=50, alpha=.7, label="x component")
    axes[1].hist(vy[valid_motion], bins=50, alpha=.7, label="y component")
    axes[1].set(title="Measured velocity components", xlabel="Velocity [m/s]", ylabel="Object observations")
    axes[1].legend()
    save_metric_figure(fig, output / "metric_velocity_histogram")
    counts, xe, ye = np.histogram2d(x, y, bins=70)
    fig, ax = metric_map_axes(x, y)
    image = ax.pcolormesh(xe, ye, counts.T, cmap="magma", shading="auto",
                           norm=LogNorm(vmin=.5, vmax=max(float(counts.max()), 1)))
    style_metric_map(ax, "World-space tracked-object centre density")
    colorbar = fig.colorbar(image, ax=ax, label="Object observations per bin (log scale)")
    colorbar.ax.tick_params(labelsize=20)
    colorbar.set_label("Object observations per bin (log scale)", fontsize=28)
    save_metric_figure(fig, output / "metric_track_center_heatmap")

    sized = [row for row in rows if row["width_m"] is not None and row["length_m"] is not None]
    if sized:
        width = np.asarray([row["width_m"] for row in sized])
        length = np.asarray([row["length_m"] for row in sized])
        long_side, short_side = np.maximum(width, length), np.minimum(width, length)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(long_side, bins=50, color="#3478bf")
        axes[0].set(title="Long-side distribution", xlabel="Long side [m]", ylabel="Object observations")
        axes[1].hist(short_side, bins=50, color="#4b9c61")
        axes[1].set(title="Short-side distribution", xlabel="Short side [m]", ylabel="Object observations")
        axes[2].hist(long_side / short_side, bins=50, color="#c87b31")
        axes[2].set(title="Aspect-ratio distribution", xlabel="Long side / short side", ylabel="Object observations")
        save_metric_figure(fig, output / "metric_object_size_distributions")

    box_count = plot_centered_oriented_boxes(
        rows, output / "metric_centered_oriented_box_distribution",
        width_key="width_m", height_key="length_m", unit="m",
        title="Metric oriented bounding-box distribution, centred at origin",
    )

    oriented = [row for row in rows if row["orientation_deg"] is not None]
    if oriented:
        ox = np.asarray([row["world_x_m"] for row in oriented])
        oy = np.asarray([row["world_y_m"] for row in oriented])
        heading = np.asarray([row["orientation_deg"] for row in oriented])
        keep = np.linspace(0, len(ox) - 1, min(len(ox), 25_000), dtype=int)
        fig, ax = metric_map_axes(ox[keep], oy[keep])
        points = ax.scatter(ox[keep], oy[keep], c=heading[keep], s=20, cmap="twilight", vmin=0, vmax=180,
                            alpha=.35, rasterized=True)
        style_metric_map(ax, "World-space position and orientation")
        colorbar = fig.colorbar(points, ax=ax, label="Orientation [deg]")
        colorbar.ax.tick_params(labelsize=24)
        colorbar.set_label("Orientation [deg]", fontsize=28)
        save_metric_figure(fig, output / "metric_position_orientation")

    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[str(row["track_id"])].append(row)
    fig, ax = metric_map_axes(x, y, colorbar=False)
    for track_id, observations in sorted(by_id.items(), key=lambda item: len(item[1]), reverse=True)[:20]:
        observations.sort(key=lambda row: row["frame"])
        ax.plot([row["world_x_m"] for row in observations], [row["world_y_m"] for row in observations], lw=1,
                alpha=.8, label=f"ID {track_id} ({len(observations)})")
    style_metric_map(ax, "Twenty longest trajectories in world space")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=7)
    save_metric_figure(fig, output / "metric_representative_trajectories")
    metric_source = "position_3d and velocity_3d in final.json" if source == "final" else "exact calibrated MVP BEV grid conversion"
    return {"metric_motion_source": metric_source, "observations": len(rows),
            "speed_mps": quantiles(speed[valid_motion]), "speed_color_cap_p95_mps": p95,
            "centred_oriented_boxes": box_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-data", type=Path, default=Path("JSON-data/final.json"),
                        help="Current tracker export; used by default for every statistic and plot.")
    parser.add_argument("--use-object-tracks", action="store_true",
                        help="Use only the legacy JSON-data/object_tracks.json BEV export instead of final.json.")
    parser.add_argument("--output", type=Path,
                        help="Base output directory. The source suffix (_final or _object_tracks) is always added.")
    parser.add_argument("--no-metric-plots", action="store_true", help="Do not create world-space metric figures.")
    args = parser.parse_args()
    source_name = "object_tracks" if args.use_object_tracks else "final"
    tracks_path = Path("JSON-data/object_tracks.json") if args.use_object_tracks else args.final_data
    output_base = args.output or Path("report_outputs/trajectories")
    args.output = output_base if output_base.name.endswith(f"_{source_name}") else output_base.parent / f"{output_base.name}_{source_name}"
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Source: {source_name} ({tracks_path})", flush=True)
    print(f"Loading track observations from {tracks_path} ...", flush=True)
    tracks = load_tracks(tracks_path)
    if not tracks:
        raise RuntimeError(f"No valid object boxes in {tracks_path}")
    write_csv(args.output / "track_observations.csv", tracks)
    plot_track_figures(tracks, args.output, None)
    plot_centered_oriented_boxes(
        tracks, args.output / "bev_centered_oriented_box_distribution",
        width_key="width_px", height_key="height_px", unit="px",
        title="BEV oriented bounding-box distribution, centred at origin",
    )

    per_track: dict[str, list[dict]] = defaultdict(list)
    frame_counts: dict[int, int] = defaultdict(int)
    for row in tracks:
        per_track[str(row["object_id"])].append(row)
        frame_counts[row["frame"]] += 1
    track_summary = []
    for object_id, observations in per_track.items():
        frames = [r["frame"] for r in observations]
        track_summary.append({"object_id": object_id, "observations": len(observations), "first_frame": min(frames),
                              "last_frame": max(frames), "median_long_side_px": float(np.median([r["long_side_px"] for r in observations]))})
    write_csv(args.output / "track_summary.csv", track_summary)
    summary = {"analysis_source": source_name, "tracks_file": str(tracks_path), "object_observations": len(tracks), "unique_track_ids": len(per_track),
               "frames_with_objects": len(frame_counts), "mean_objects_per_nonempty_frame": float(np.mean(list(frame_counts.values()))),
               "long_side_px": quantiles(r["long_side_px"] for r in tracks),
               "short_side_px": quantiles(r["short_side_px"] for r in tracks),
               "area_px2": quantiles(r["area_px2"] for r in tracks)}
    speeds = [r["speed_px_per_frame"] for r in tracks if np.isfinite(r["speed_px_per_frame"])]
    if speeds:
        summary["estimated_speed_px_per_frame"] = quantiles(speeds)
    if not args.no_metric_plots:
        if args.use_object_tracks:
            metric_rows = metric_motion_from_bev_tracks(tracks)
        else:
            print(f"Loading direct metric motion from {args.final_data} ...", flush=True)
            metric_rows = load_metric_motion(args.final_data)
        if metric_rows:
            write_csv(args.output / "metric_motion_observations.csv", metric_rows)
            summary["metric_motion"] = plot_metric_motion(metric_rows, args.output, source_name)
    with (args.output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote trajectory statistics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
