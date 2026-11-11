#!/usr/bin/env python3
"""Interactively define a factory intersection polygon on a BEV image.

Left-click to add a vertex, right-click to remove the last vertex, then press
S or Enter to save. The saved ground coordinates assume the factory floor is
the z=0 plane of ``occupancy_projection``.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "Select intersection area"


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_configuration():
    with (REPO_ROOT / "mvp-config.json").open() as file:
        config = json.load(file)

    dataset_root = repo_path(config["paths"]["dataset_root"])
    image_dir = dataset_root / "bev_img"
    projection_path = repo_path(config["extra_paths"]["occupancy_projection"])
    return image_dir, projection_path


def image_to_ground(points_px: List[Tuple[int, int]], projection: np.ndarray):
    """Back-project BEV pixels to ground-plane coordinates (z=0)."""
    # For z=0, P [X, Y, 0, 1]^T is a 2D homography from floor to BEV pixels.
    homography = projection[:, [0, 1, 3]]
    inverse_homography = np.linalg.inv(homography)
    ground_points = []
    for u, v in points_px:
        ground_h = inverse_homography @ np.array([u, v, 1.0])
        x, y = (ground_h[:2] / ground_h[2]).tolist()
        ground_points.append([x, y, 0.0])
    return ground_points


def draw(image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    canvas = image.copy()
    for index, point in enumerate(points):
        cv2.circle(canvas, point, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, str(index + 1), (point[0] + 7, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA,
        )
    if len(points) >= 2:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False,
                      (0, 255, 255), 2, cv2.LINE_AA)
    if len(points) >= 3:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], True,
                      (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Left click: add | Right click: undo | S/Enter: save | Esc: quit",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Left click: add | Right click: undo | S/Enter: save | Esc: quit",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
        cv2.LINE_AA,
    )
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="BEV image to use (default: first image in bev_img)")
    parser.add_argument(
        "--output", type=Path, default=Path("JSON-data/intersection_area.json"),
        help="JSON output path, relative to the repository by default",
    )
    args = parser.parse_args()

    image_dir, projection_path = load_configuration()
    image_path = args.image or next(iter(sorted(image_dir.glob("*.jpg"))), None)
    if image_path is None:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")
    if not image_path.is_absolute():
        image_path = repo_path(str(image_path))

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")

    with projection_path.open("rb") as file:
        projection_data = pickle.load(file)
    projection = np.asarray(projection_data["P"], dtype=np.float64)
    if projection.shape != (3, 4):
        raise ValueError(f"Expected a 3x4 projection matrix, got {projection.shape}")

    points: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    while True:
        cv2.imshow(WINDOW_NAME, draw(image, points))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return
        if key in (ord("s"), ord("S"), 13, 10):
            if len(points) < 3:
                print("Select at least three vertices before saving.")
                continue
            break

    output_path = args.output if args.output.is_absolute() else repo_path(str(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ground_points = image_to_ground(points, projection)
    payload = {
        "source_image": str(image_path.relative_to(REPO_ROOT)),
        "image_size_px": {"width": image.shape[1], "height": image.shape[0]},
        "polygon_image_px": [[x, y] for x, y in points],
        "polygon_ground_world_m": ground_points,
        "ground_plane": "z=0",
    }
    with output_path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
    cv2.imwrite(str(overlay_path), draw(image, points))
    cv2.destroyAllWindows()
    print(f"Saved coordinates: {output_path}")
    print(f"Saved polygon preview: {overlay_path}")


if __name__ == "__main__":
    main()
