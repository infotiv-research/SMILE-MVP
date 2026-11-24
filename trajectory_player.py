from __future__ import annotations

import argparse, json, math, os, pickle, sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ==========================================================
# CONFIG + BEV BACKGROUND
# ==========================================================
CONFIG_FILE = "mvp-config.json"
with open(CONFIG_FILE) as f:
    CFG = json.load(f)

DATA_ROOT = Path(CFG["paths"]["dataset_root"])
DATA_PATH = str(DATA_ROOT / CFG["paths"]["save_path"])
EXTRA_PATHS = CFG["extra_paths"]
OCCUPANCY_PROJECTION = str(DATA_ROOT / EXTRA_PATHS["occupancy_projection"])
BEV_FOLDER = str(DATA_ROOT / EXTRA_PATHS["mask_folder"])

EXPORT_FAST_MODE = True

bev_lookup = {}
if os.path.isdir(BEV_FOLDER):
    _bev_files = os.listdir(BEV_FOLDER)
else:
    _bev_files = []
    print(f"[WARN] BEV folder not found: {BEV_FOLDER}")
for f in _bev_files:
    if f.lower().endswith(".jpg"):
        try:
            bev_lookup[int(Path(f).stem)] = f
        except ValueError:
            pass

# --------------------- projection helpers ---------------------
def load_occupancy_projection():
    with open(OCCUPANCY_PROJECTION, "rb") as f:
        return np.asarray(pickle.load(f)["P"], dtype=np.float64)

P_OCC = None

def get_projection():
    global P_OCC
    if P_OCC is None:
        try:
            P_OCC = load_occupancy_projection()
        except Exception:
            P_OCC = None
    return P_OCC

def world_to_occupancy(wx, wy):
    P = get_projection()
    if P is None:
        return wx, wy
    uvw = (P @ np.array([[wx], [wy], [0.0], [1.0]])).ravel()
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])

def world_poly_to_occ(poly):
    return np.asarray([world_to_occupancy(x, y) for x, y in np.asarray(poly)], dtype=float)

# --------------------- grid smoothing ---------------------
TRAJ_GRID = 0.25  # meters

def smooth_trajectory(arr, grid=TRAJ_GRID):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 7:
        return arr
    win = min(11, len(arr))
    win -= (1 - win % 2)
    xs = savgol_filter(arr[:, 0], win, 2)
    ys = savgol_filter(arr[:, 1], win, 2)
    sm = np.round(np.column_stack([xs, ys]) / grid) * grid
    keep = [0]
    for i in range(1, len(sm)):
        if not np.allclose(sm[i], sm[keep[-1]]):
            keep.append(i)
    return sm[keep]

# --------------------- model storage ---------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def models_dir() -> str:
    d = os.path.join(os.getcwd(), "models")
    ensure_dir(d)
    return d

def timestamped_model_path() -> str:
    return os.path.join(models_dir(), f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")

def list_saved_models() -> List[str]:
    d = models_dir()
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files

def get_model_entry_count(path):
    """Number of entries stored in a saved model (old and new file formats)."""
    try:
        with np.load(path, allow_pickle=True) as d:
            if "entry_centers" in d:
                return len(d["entry_centers"])
    except Exception:
        pass
    return None

# --------------------- geometry ---------------------
def point_in_poly(pt, poly):
    x, y = pt[0], pt[1]
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def _point_to_segment_distance(pt, a, b):
    pt, a, b = map(lambda x: np.asarray(x, dtype=float), (pt, a, b))
    ab = b - a
    t = np.clip(np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(pt - (a + t * ab)))

def _point_to_poly_distance(pt, poly):
    poly = np.asarray(poly, dtype=float)
    if point_in_poly(pt, poly):
        return 0.0
    return min(_point_to_segment_distance(pt, poly[i], poly[(i + 1) % len(poly)])
               for i in range(len(poly)))

def classify_topology_state(model, pos):
    """Classify by polygon containment first, boundary distance second."""
    if model.intersection_poly is None or not model.zone_polys:
        return {"zone": None, "edge": None, "distance": np.inf}
    if point_in_poly(pos, model.intersection_poly):
        return {"zone": "I", "edge": "I", "distance": 0.0}
    containing = [z for z, poly in model.zone_polys.items() if point_in_poly(pos, poly)]
    candidates = containing or list(model.zone_polys)
    best = min(candidates, key=lambda z: _point_to_poly_distance(pos, model.zone_polys[z]))
    return {"zone": best, "edge": str(best),
            "distance": _point_to_poly_distance(pos, model.zone_polys[best])}

def _assign_entry_labels(trajs, entry_centers, entry_match_radius=4.0):
    """Valid (tid, start_entry, end_entry, trajectory) tuples."""
    labelled = []
    entry_centers = np.asarray(entry_centers, dtype=float)
    for tid, arr in trajs.items():
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 6:
            continue
        sd = np.linalg.norm(entry_centers - arr[0], axis=1)
        ed = np.linalg.norm(entry_centers - arr[-1], axis=1)
        sidx, eidx = int(np.argmin(sd)), int(np.argmin(ed))
        if sd[sidx] > entry_match_radius or ed[eidx] > entry_match_radius or sidx == eidx:
            continue
        labelled.append((tid, sidx, eidx, arr))
    return labelled

def _cluster_entry_arms(entry_centers, n_arms=3, seed=0):
    """Group nearby entry areas into physical corridor arms."""
    entry_centers = np.asarray(entry_centers, dtype=float)
    k = max(2, min(int(n_arms), len(entry_centers)))
    # Standardization prevents the long X range from hiding vertical separation.
    z = (entry_centers - entry_centers.mean(axis=0)) / (entry_centers.std(axis=0) + 1e-9)
    return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(z)

def _ray_box_boundary_point(origin, target, poly):
    """Intersection of origin->target ray with an axis-aligned polygon boundary."""
    origin, target = np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
    d = target - origin
    xmin, xmax = np.min(poly[:, 0]), np.max(poly[:, 0])
    ymin, ymax = np.min(poly[:, 1]), np.max(poly[:, 1])
    hits = []
    if abs(d[0]) > 1e-12:
        for x in (xmin, xmax):
            t = (x - origin[0]) / d[0]
            y = origin[1] + t * d[1]
            if 0.0 <= t <= 1.0 and ymin - 1e-9 <= y <= ymax + 1e-9:
                hits.append((t, np.array([x, y])))
    if abs(d[1]) > 1e-12:
        for y in (ymin, ymax):
            t = (y - origin[1]) / d[1]
            x = origin[0] + t * d[0]
            if 0.0 <= t <= 1.0 and xmin - 1e-9 <= x <= xmax + 1e-9:
                hits.append((t, np.array([x, y])))
    return min(hits, key=lambda z: z[0])[1] if hits else target

def detect_intersection_region(trajs, entry_centers, entry_match_radius=4.0):
    """Junction = smallest area where cross-arm trajectories become close (WORLD coords)."""
    labelled = _assign_entry_labels(trajs, entry_centers, entry_match_radius)
    arms = _cluster_entry_arms(entry_centers, n_arms=3)
    eligible = [arr for _, s, e, arr in labelled if arms[s] != arms[e]]
    if len(eligible) < 3:
        raise RuntimeError("Too few cross-arm trajectories.")
    # Iteratively locate the common crossing.
    center = np.median(np.vstack([resample_polyline(t, 51)[25] for t in eligible]), axis=0)
    for _ in range(20):
        anchors = np.asarray([arr[np.argmin(np.linalg.norm(arr - center, axis=1))]
                              for arr in eligible])
        new_center = np.median(anchors, axis=0)
        if np.linalg.norm(new_center - center) < 1e-3:
            break
        center = new_center
    # Junction size from the anchor cloud.
    dx = np.clip(np.percentile(np.abs(anchors[:, 0] - center[0]), 75) + 0.5, 1.5, 3.0)
    dy = np.clip(np.percentile(np.abs(anchors[:, 1] - center[1]), 75) + 0.5, 1.5, 3.0)
    def rect():
        return np.array([[center[0] - dx, center[1] - dy], [center[0] + dx, center[1] - dy],
                         [center[0] + dx, center[1] + dy], [center[0] - dx, center[1] + dy]])
    poly = rect()
    # Never contain entry centers.
    for ec in entry_centers:
        while point_in_poly(ec, poly):
            dx, dy = dx * 0.95, dy * 0.95
            poly = rect()
    return poly

def build_branch_zones(entry_centers, intersection_poly, trajs):
    """Corridor polygons from actual trajectory approach segments, stopping at the junction."""
    center = intersection_poly.mean(axis=0)
    labelled = _assign_entry_labels(trajs, entry_centers)
    zones = {}
    for zid, entry in enumerate(entry_centers):
        approach_pts = []
        for _, sidx, eidx, arr in labelled:
            if zid not in (sidx, eidx):
                continue
            n = max(10, len(arr) // 4)
            approach_pts.append(arr[:n] if sidx == zid else arr[-n:])
        if not approach_pts:
            continue
        pts = np.vstack(approach_pts)
        # Dominant lane direction.
        vals, vecs = np.linalg.eigh(np.cov(pts.T))
        v = vecs[:, int(np.argmax(vals))]
        if np.dot(center - entry, v) < 0:
            v = -v
        nvec = np.array([-v[1], v[0]])
        # Corridor width.
        width = np.clip(np.percentile(np.abs((pts - entry) @ nvec), 95) + 0.25, 1.5, 3.5)
        end_pt = _ray_box_boundary_point(entry, center, intersection_poly)
        zones[zid] = np.array([entry + width * nvec, entry - width * nvec,
                               end_pt - width * nvec, end_pt + width * nvec])
    return zones

def collect_terminal_points(trajs, tail_len=15, max_terminal_radius=2.5):
    terminals = []
    for arr in trajs.values():
        if len(arr) < tail_len:
            continue
        for seg in (arr[:tail_len], arr[-tail_len:]):
            c = seg.mean(axis=0)
            if np.max(np.linalg.norm(seg - c, axis=1)) < max_terminal_radius:
                terminals.append(c)
    return np.asarray(terminals)

# --------------------- data loaders ---------------------
def load_trajectories(json_path: str) -> Dict[int, np.ndarray]:
    """(A) {tid: [[x,y],...]} or (B) frame-based list -> {tid: ndarray(N,2)}."""
    with open(json_path) as f:
        raw = json.load(f)
    bucket: Dict[int, np.ndarray] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                tid = int(k)
            except Exception:
                continue
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
                bucket[tid] = arr
    elif isinstance(raw, list):
        for frame in raw:
            for obj in frame.get("object_list", []):
                tid, pos = obj.get("track_id"), obj.get("position_3d")
                if tid is None or pos is None:
                    continue
                x, y = pos.get("x"), pos.get("y")
                if x is None or y is None:
                    continue
                bucket.setdefault(int(tid), []).append([float(x), float(y)])
    else:
        raise RuntimeError("Unrecognized JSON trajectory format.")
    trajs: Dict[int, np.ndarray] = {}
    for tid, pts in bucket.items():
        arr = smooth_trajectory(np.asarray(pts, dtype=float), grid=TRAJ_GRID)
        if len(arr) >= 3:
            trajs[tid] = arr
    if not trajs:
        raise RuntimeError("No valid trajectories found in JSON.")
    return trajs

def load_frame_based(json_path: str):
    """Strict frame-based loader -> (frame_list, frame_objects, tid_frame_history, raw)."""
    with open(json_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("This file is not in frame-based JSON format (expected a list).")
    frame_list: List[int] = []
    frame_objects: List[Dict[int, List[float]]] = []
    tid_frame_history: Dict[int, List[Tuple[int, List[float]]]] = {}
    for i, frame in enumerate(data):
        objs: Dict[int, List[float]] = {}
        for o in frame.get("object_list", []):
            tid, pos = o.get("track_id"), o.get("position_3d", {})
            if tid is None:
                continue
            x, y = pos.get("x"), pos.get("y")
            if x is None or y is None:
                continue
            objs[int(tid)] = [float(x), float(y)]
            tid_frame_history.setdefault(int(tid), []).append((i, [float(x), float(y)]))
        frame_list.append(frame.get("time_stamp"))
        frame_objects.append(objs)
    return frame_list, frame_objects, tid_frame_history, data

# --------------------- resampling & ellipses ---------------------
def resample_polyline(xy: np.ndarray, m: int = 32) -> np.ndarray:
    """Resample a polyline to m points by arc length."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) <= 1:
        return np.repeat(xy[:1], m, axis=0)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(d)]
    if s[-1] < 1e-8:
        return np.repeat(xy[:1], m, axis=0)
    tgt = np.linspace(0, s[-1], m)
    out = np.zeros((m, 2))
    j = 0
    for i, si in enumerate(tgt):
        while j + 1 < len(s) and s[j + 1] < si:
            j += 1
        if j + 1 == len(s):
            j = len(s) - 2
        t = (si - s[j]) / (s[j + 1] - s[j] + 1e-12)
        out[i] = xy[j] + t * (xy[j + 1] - xy[j])
    return out

def ellipse_params(points: np.ndarray, n_std: float = 2.5):
    """((cx,cy), width, height, angle_deg) for a Gaussian ellipse."""
    c = points.mean(axis=0)
    cov = np.cov(points.T) if points.shape[0] > 1 else np.eye(2) * 1e-3
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return (c[0], c[1]), w, h, ang

def ellipse_poly(center, width, height, angle_deg, n=200):
    cx, cy = center
    t = np.linspace(0, 2 * np.pi, n)
    ca, sa = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    R = np.array([[ca, -sa], [sa, ca]])
    return np.c_[width / 2 * np.cos(t), height / 2 * np.sin(t)] @ R.T + np.array([cx, cy])

# ==========================================================
# ML Model (features, predict, save/load)
# ==========================================================
class MLModel:
    """Route classifier (start_entry -> end_entry) with learned features
    and progress-aware dynamics."""

    def __init__(self):
        self.entry_centers: Optional[np.ndarray] = None          # (K,2)
        self.entry_ellipses: Optional[List[Tuple]] = None        # [(center,w,h,ang), ...]
        self.pair_to_cid: Dict[Tuple[int, int], int] = {}
        self.cid_to_pair: Dict[int, Tuple[int, int]] = {}
        self.code_centers: Optional[np.ndarray] = None           # (K_code,2)
        self.pca: Optional[PCA] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.clf: Optional[LogisticRegression] = None
        self.class_step_stats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.progress_bins: Optional[np.ndarray] = None
        self.class_progress_stats: Dict[int, Dict[str, np.ndarray]] = {}
        self.intersection_poly = None
        self.zone_polys = None

    # ---- features ----
    def _code_hist(self, dif):
        if dif.size == 0:
            return np.zeros(self.code_centers.shape[0], dtype=float)
        d2 = ((dif[:, None, :] - self.code_centers[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        hist = np.bincount(idx, minlength=self.code_centers.shape[0]).astype(float)
        return hist / (hist.sum() + 1e-12)

    def _shape_pca(self, seg):
        R = resample_polyline(seg, m=32)
        R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-6)
        return self.pca.transform(R.reshape(1, -1))[0]

    def seg_features(self, seg):
        dif = np.diff(seg, axis=0)
        spd = np.linalg.norm(dif, axis=1)
        path_len = float(spd.sum()) if len(spd) else 0.0
        net = float(np.linalg.norm(seg[-1] - seg[0]))
        if len(dif) >= 2:
            th = np.unwrap(np.arctan2(dif[:, 1], dif[:, 0]))
            rev_frac = float(np.mean(np.abs(np.diff(th)) > np.deg2rad(120)))
        else:
            rev_frac = 0.0
        centroid = seg.mean(axis=0)
        return np.r_[
            [path_len, net, net / (path_len + 1e-12),
             float(spd.mean()) if len(spd) else 0.0, float(spd.std()) if len(spd) else 0.0,
             rev_frac,
             float(np.sqrt(np.mean(((seg - centroid) ** 2).sum(axis=1)))),
             float(np.linalg.norm(seg.max(axis=0) - seg.min(axis=0)))],
            np.linalg.norm(self.entry_centers - seg[-1], axis=1),
            np.linalg.norm(self.entry_centers - seg[0], axis=1),
            self._code_hist(dif),
            self._shape_pca(seg),
        ]

    # ---- predict ----
    def predict_segment_ml(self, seg, topk=3):
        xz = (self.seg_features(seg) - self.mu) / self.sigma
        p = self.clf.predict_proba(xz[None, :])[0]
        order = np.argsort(-p)
        top = order[:topk]
        return ([self.cid_to_pair[int(i)] for i in top],
                [float(p[i]) for i in top],
                float(1.0 - p[order[0]]))

    # ---- save / load ----
    def save(self, path):
        ensure_dir(os.path.dirname(path))
        C = len(self.cid_to_pair)
        cid_to_pair = np.array([self.cid_to_pair[i] for i in range(C)], dtype=int)
        mus = np.zeros((C, 2))
        covs = np.zeros((C, 2, 2))
        for i in range(C):
            m, S = self.class_step_stats.get(i, (np.zeros(2), np.eye(2) * 1e-3))
            mus[i], covs[i] = m, S
        ell = np.array([[e[0][0], e[0][1], e[1], e[2], e[3]] for e in self.entry_ellipses],
                       dtype=float)
        if self.progress_bins is None:
            self.progress_bins = np.linspace(0.05, 0.95, 10)
        B = len(self.progress_bins)
        pmus = np.zeros((C, B, 2))
        pcovs = np.zeros((C, B, 2, 2))
        pcounts = np.zeros((C, B))
        avg_len = np.zeros(C)
        for i in range(C):
            st = self.class_progress_stats.get(i)
            if st:
                pmus[i], pcovs[i], pcounts[i], avg_len[i] = st["mus"], st["covs"], st["counts"], st["avg_len"]
            else:
                pcovs[i] = np.repeat(np.eye(2)[None, :, :], B, axis=0)
                avg_len[i] = 1.0
        zone_ids = np.array(sorted((self.zone_polys or {}).keys()), dtype=int)
        zone_polys = (np.stack([self.zone_polys[i] for i in zone_ids]) if len(zone_ids)
                      else np.empty((0, 4, 2), dtype=float))
        intersection_poly = (np.asarray(self.intersection_poly, dtype=float)
                             if self.intersection_poly is not None else np.empty((0, 2), dtype=float))
        np.savez_compressed(
            path,
            entry_centers=self.entry_centers,
            intersection_poly=intersection_poly,
            zone_ids=zone_ids,
            zone_polys=zone_polys,
            entry_ellipses=ell,
            code_centers=self.code_centers,
            pca_mean=self.pca.mean_,
            pca_components=self.pca.components_,
            pca_explained_variance=self.pca.explained_variance_,
            pca_explained_variance_ratio=self.pca.explained_variance_ratio_,
            pca_singular_values=self.pca.singular_values_,
            pca_whiten=self.pca.whiten,
            pca_n=self.pca.n_components_,
            mu=self.mu,
            sigma=self.sigma,
            clf_coef=self.clf.coef_,
            clf_inter=self.clf.intercept_,
            clf_classes=self.clf.classes_,
            cid_to_pair=cid_to_pair,
            step_mus=mus,
            step_covs=covs,
            progress_bins=self.progress_bins,
            prog_mus=pmus,
            prog_covs=pcovs,
            prog_counts=pcounts,
            class_avg_len=avg_len,
        )

    @staticmethod
    def load(path) -> "MLModel":
        z = np.load(path, allow_pickle=False)
        m = MLModel()
        m.entry_centers = z["entry_centers"]
        if "intersection_poly" in z.files and len(z["intersection_poly"]) >= 3:
            m.intersection_poly = z["intersection_poly"]
        if "zone_ids" in z.files and "zone_polys" in z.files:
            m.zone_polys = {int(i): p for i, p in zip(z["zone_ids"], z["zone_polys"])}
        m.entry_ellipses = [((r[0], r[1]), r[2], r[3], r[4]) for r in z["entry_ellipses"]]
        m.code_centers = z["code_centers"]
        pca = PCA(n_components=int(z["pca_n"]), whiten=bool(z["pca_whiten"]))
        pca.mean_ = z["pca_mean"]
        pca.components_ = z["pca_components"]
        pca.explained_variance_ = z["pca_explained_variance"]       # required by transform()
        pca.explained_variance_ratio_ = z["pca_explained_variance_ratio"]
        pca.singular_values_ = z["pca_singular_values"]
        m.pca = pca
        m.mu, m.sigma = z["mu"], z["sigma"]
        clf = LogisticRegression(solver="lbfgs", max_iter=1)        # dummy; weights restored below
        clf.classes_ = z["clf_classes"]
        clf.coef_ = z["clf_coef"]
        clf.intercept_ = z["clf_inter"]
        clf.n_features_in_ = clf.coef_.shape[1]
        clf.n_iter_ = np.array([1], dtype=np.int32)
        m.clf = clf
        cid_to_pair = z["cid_to_pair"]
        m.cid_to_pair = {int(i): (int(cid_to_pair[i, 0]), int(cid_to_pair[i, 1]))
                         for i in range(cid_to_pair.shape[0])}
        m.pair_to_cid = {v: k for k, v in m.cid_to_pair.items()}
        for i in range(z["step_mus"].shape[0]):
            m.class_step_stats[i] = (z["step_mus"][i], z["step_covs"][i])
        m.progress_bins = z["progress_bins"]
        for i in range(z["prog_mus"].shape[0]):
            m.class_progress_stats[i] = {"mus": z["prog_mus"][i], "covs": z["prog_covs"][i],
                                         "counts": z["prog_counts"][i],
                                         "avg_len": float(z["class_avg_len"][i])}
        return m

def ensure_model_topology(model, trajs):
    """Populate topology for legacy model files predating topology persistence."""
    if model.intersection_poly is None or not model.zone_polys:
        model.intersection_poly = detect_intersection_region(trajs, model.entry_centers)
        model.zone_polys = build_branch_zones(model.entry_centers, model.intersection_poly, trajs)
    return model

# ==========================================================
# Training pipeline
# ==========================================================
def auto_tune_and_train_ml(trajs, K_ENTRIES=7, K_CODE=16, N_PCA=6,
                           prefix_fracs=(0.15, 0.25, 0.35, 0.5, 0.7, 0.9), seed=0):
    model = MLModel()
    # 1) Entry areas: cluster terminal points into corridor entries.
    P = collect_terminal_points(trajs, tail_len=15, max_terminal_radius=2.5)
    if len(P) < K_ENTRIES:
        raise RuntimeError(f"Only {len(P)} terminal samples found.")
    km = KMeans(n_clusters=K_ENTRIES, random_state=seed, n_init="auto").fit(P)
    labels, centers = km.labels_, km.cluster_centers_
    raw_ellipses, keep = [], []
    for ci in range(K_ENTRIES):
        pts = P[labels == ci]
        if len(pts) < 3:
            continue
        center, w, h, ang = ellipse_params(pts, n_std=2.0)
        print(f"[ENTRY {ci}] center=({center[0]:.2f},{center[1]:.2f}) n={len(pts)} "
              f"w={w:.2f} h={h:.2f} ratio={max(w, h) / max(1e-6, min(w, h)):.2f} area={w * h:.2f}")
        # (aspect-ratio rejection of non-corridor clusters is currently disabled)
        raw_ellipses.append((ci, center, w, h, ang))
        keep.append(ci)
    if len(keep) < 2:
        raise RuntimeError("Too few valid entry clusters remain.")
    # Stable left->right numbering.
    order = np.argsort(np.array([centers[ci] for ci in keep])[:, 0])
    old_to_new = {}
    entry_centers, entry_ellipses = [], []
    for new_id, idx in enumerate(order):
        old_id = keep[idx]
        old_to_new[old_id] = new_id
        _, center, w, h, ang = raw_ellipses[idx]
        entry_centers.append(centers[old_id])
        entry_ellipses.append((center, w, h, ang))
    model.entry_centers = np.asarray(entry_centers, dtype=float)
    model.entry_ellipses = entry_ellipses
    model.intersection_poly = detect_intersection_region(trajs, model.entry_centers)
    model.zone_polys = build_branch_zones(model.entry_centers, model.intersection_poly, trajs)
    model.old_to_new_entry = old_to_new
    print(f"[INFO] Using {len(entry_centers)} entries")

    # 2) Route labels.
    traj_pair = {}
    for tid, arr in trajs.items():
        sd = np.linalg.norm(centers - arr[0], axis=1)
        ed = np.linalg.norm(centers - arr[-1], axis=1)
        sc_old, ec_old = int(np.argmin(sd)), int(np.argmin(ed))
        if sc_old not in old_to_new or ec_old not in old_to_new:
            continue
        if sd[sc_old] > 4.0 or ed[ec_old] > 4.0:
            continue
        traj_pair[tid] = (old_to_new[sc_old], old_to_new[ec_old])
    if len(traj_pair) < 2:
        raise RuntimeError("Too few valid trajectories.")
    model.pair_to_cid = {p: i for i, p in enumerate(sorted(set(traj_pair.values())))}
    model.cid_to_pair = {i: p for p, i in model.pair_to_cid.items()}

    # 3) Codebook over step vectors.
    steps_list = [s for s in (np.diff(arr, axis=0) for arr in trajs.values() if len(arr) >= 2) if len(s)]
    steps = np.vstack(steps_list) if steps_list else np.zeros((1, 2))
    sp = np.linalg.norm(steps, axis=1)
    lo, hi = (np.quantile(sp, [0.01, 0.99]) if len(sp) > 10 else (sp.min(), sp.max() + 1e-6))
    base = steps[(sp >= lo) & (sp <= hi)] if len(sp) > 10 else steps
    model.code_centers = KMeans(n_clusters=K_CODE, random_state=seed, n_init="auto").fit(base).cluster_centers_

    # 4) PCA on shapes.
    resampled = []
    for arr in list(trajs.values())[:min(120, len(trajs))]:
        R = resample_polyline(arr, m=32)
        resampled.append(((R - R.mean(axis=0)) / (R.std(axis=0) + 1e-6)).reshape(-1))
    Xp = np.vstack(resampled) if resampled else np.zeros((1, 64))
    model.pca = PCA(n_components=N_PCA, whiten=False, random_state=seed).fit(Xp)

    # 5) Prefix training set.
    X_list, y_list = [], []
    for tid, arr in trajs.items():
        if tid not in traj_pair:
            continue
        cid = model.pair_to_cid[traj_pair[tid]]
        for f in prefix_fracs:
            n = max(5, int(len(arr) * f))
            if n >= len(arr):
                n = len(arr) - 1
            if n <= 4:
                continue
            X_list.append(model.seg_features(arr[:n]))
            y_list.append(cid)
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    model.mu = X.mean(axis=0)
    model.sigma = X.std(axis=0) + 1e-6

    # 6) Multinomial logistic regression.
    clf = LogisticRegression(solver="lbfgs", max_iter=400, random_state=seed)
    clf.fit((X - model.mu) / model.sigma, y)
    model.clf = clf

    # 7) Per-class step dynamics.
    class_stats = {}
    for tid, arr in trajs.items():
        if tid in traj_pair:
            class_stats.setdefault(model.pair_to_cid[traj_pair[tid]], []).append(np.diff(arr, axis=0))
    for cid, lst in class_stats.items():
        V = np.vstack(lst)
        model.class_step_stats[cid] = (V.mean(axis=0),
                                       np.cov(V.T) if V.shape[0] > 1 else np.eye(2) * 1e-3)

    # 8) Progress-aware dynamics (10 bins).
    B = 10
    model.progress_bins = np.linspace(0.05, 0.95, B)
    prog = {cid: {"sum": np.zeros((B, 2)), "cross": np.zeros((B, 2, 2)),
                  "count": np.zeros(B), "lens": []} for cid in model.cid_to_pair}
    for tid, arr in trajs.items():
        if tid not in traj_pair:
            continue
        cid = model.pair_to_cid[traj_pair[tid]]
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        if not len(d):
            continue
        s = np.r_[0, np.cumsum(d)]
        if s[-1] < 1e-8:
            continue
        dif = np.diff(arr, axis=0)
        for i in range(len(dif)):
            b = int(np.clip(np.round((B - 1) * s[i] / s[-1]), 0, B - 1))
            prog[cid]["sum"][b] += dif[i]
            prog[cid]["cross"][b] += np.outer(dif[i], dif[i])
            prog[cid]["count"][b] += 1
        prog[cid]["lens"].append(s[-1])
    for cid in model.cid_to_pair:
        cnt = prog[cid]["count"]
        mus = np.zeros((B, 2))
        covs = np.zeros((B, 2, 2))
        for b in range(B):
            if cnt[b] > 1:
                mu_b = prog[cid]["sum"][b] / cnt[b]
                mus[b] = mu_b
                covs[b] = prog[cid]["cross"][b] / cnt[b] - np.outer(mu_b, mu_b) + 1e-6 * np.eye(2)
            else:
                covs[b] = np.eye(2) * 1e-3
        model.class_progress_stats[cid] = {
            "mus": mus, "covs": covs, "counts": cnt,
            "avg_len": float(np.mean(prog[cid]["lens"]) if prog[cid]["lens"] else 1.0),
        }
    return model

def evaluate_prefix_accuracy(trajs, model, prefix_fracs=(0.15, 0.25, 0.35, 0.5, 0.7, 0.9), topk=3):
    true_pair = {}
    for tid, arr in trajs.items():
        sc = int(np.argmin(np.linalg.norm(model.entry_centers - arr[0], axis=1)))
        ec = int(np.argmin(np.linalg.norm(model.entry_centers - arr[-1], axis=1)))
        if (sc, ec) in model.pair_to_cid:      # skip pairs unseen in training
            true_pair[tid] = (sc, ec)
    results = []
    for f in prefix_fracs:
        c1 = cK = tot = 0
        for tid, arr in trajs.items():
            if tid not in true_pair:
                continue
            n = min(max(6, int(len(arr) * f)), len(arr) - 1)
            if n <= 5:
                continue
            classes, _, _ = model.predict_segment_ml(arr[:n], topk=topk)
            t = true_pair[tid]
            tot += 1
            if classes[0] == t:
                c1 += 1
            if t in classes:
                cK += 1
        if tot > 0:
            results.append((f, c1 / tot, cK / tot, tot))
    return results

def save_accuracy_plot(results, out_path):
    ensure_dir(os.path.dirname(out_path))
    fracs = [r[0] * 100 for r in results]
    top1 = [r[1] * 100 for r in results]
    topk = [r[2] * 100 for r in results]
    n = [r[3] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, top1, marker="o", label="Top-1")
    ax.plot(fracs, topk, marker="s", label="Top-3")
    for x, y, c in zip(fracs, top1, n):
        ax.text(x, y + 1.5, f"n={c}", ha="center", fontsize=8)
    ax.set_xlabel("Prefix length (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Route accuracy vs prefix")
    ax.set_ylim(0, 105)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# ==========================================================
# Micro maneuvers
# ==========================================================
def compute_window_features(seg):
    seg = np.asarray(seg, dtype=float)
    if len(seg) < 3:
        return {"path_len": 0.0, "net_disp": 0.0, "efficiency": 0.0, "area": 0.0,
                "median_speed": 0.0, "speed_p95": 0.0, "heading_std": 0.0, "path_ratio": 0.0}
    dif = np.diff(seg, axis=0)
    speed = np.linalg.norm(dif, axis=1)
    path_len = float(speed.sum())
    net_disp = float(np.linalg.norm(seg[-1] - seg[0]))
    efficiency = net_disp / max(path_len, 1e-6)
    cov = np.cov((seg - seg.mean(axis=0)).T)
    eig = np.maximum(np.linalg.eigvalsh(cov), 1e-8)
    heading = np.unwrap(np.arctan2(dif[:, 1], dif[:, 0]))
    return {"path_len": path_len, "net_disp": net_disp, "efficiency": efficiency,
            "area": float(np.pi * np.sqrt(eig[0]) * np.sqrt(eig[1])),
            "median_speed": float(np.median(speed)),
            "speed_p95": float(np.percentile(speed, 95)),
            "heading_std": float(np.std(np.diff(heading))) if len(heading) > 1 else 0.0,
            "path_ratio": path_len / max(net_disp, 1e-6)}

def auto_tune_mode_thresholds(trajs, win=25, rev_angle_deg=120.0):
    """Reserved for future threshold auto-tuning; result currently unused."""
    return {}

def motion_mode_probs(history_seg, recent_seg=None):
    """Forklift manoeuvre classifier: moving / loading / stationary."""
    fh = compute_window_features(history_seg)
    fr = compute_window_features(recent_seg if recent_seg is not None else history_seg)
    path_len = 0.3 * fh["path_len"] + 0.7 * fr["path_len"]
    net_disp = 0.3 * fh["net_disp"] + 0.7 * fr["net_disp"]
    heading_std = 0.3 * fh["heading_std"] + 0.7 * fr["heading_std"]
    area = 0.3 * fh["area"] + 0.7 * fr["area"]
    efficiency = 0.3 * fh["efficiency"] + 0.7 * fr["efficiency"]
    wandering = path_len / max(net_disp, 0.2)                 # key loading indicator
    stationary = np.exp(-4.0 * path_len)
    loading = (np.tanh(path_len / 2.0) * (1.0 - efficiency)
               * np.tanh(area / 2.0) * np.tanh(wandering / 3.0))
    moving = np.tanh(net_disp / 3.0) * efficiency * np.exp(-0.3 * heading_std)
    s = max(stationary, 0.0) + max(loading, 0.0) + max(moving, 0.0) + 1e-12
    return {"stationary": max(stationary, 0.0) / s,
            "loading": max(loading, 0.0) / s,
            "moving": max(moving, 0.0) / s}

class OnlineClassifier:
    """Route + micro-mode classifier with belief decay and topology-aware labels.

    NOTE: enter/exit hysteresis (enter_baf/exit_baf) and the *_sens parameters are
    accepted for API compatibility but not applied yet.
    """

    def __init__(self, model, T, window=25, enter_baf=3, exit_baf=5, rev_angle_deg=120.0,
                 baf_sensitivity=1.0, stationary_sens=1.0, bbox_sens=1.0):
        self.model = model
        self.T = T
        self.window = window
        self.enter_baf = enter_baf
        self.exit_baf = exit_baf
        self.rev_angle_deg = rev_angle_deg
        self.baf_sensitivity = baf_sensitivity
        self.stationary_sens = stationary_sens
        self.bbox_sens = bbox_sens
        self.mode = "moving"
        self.route_belief = {}
        self._last_prediction_signature = None
        self._cached_info = None

    def update(self, prefix):
        sig = (len(prefix), round(float(prefix[-1, 0]), 2), round(float(prefix[-1, 1]), 2))
        if self._cached_info is not None and self._last_prediction_signature == sig:
            return self._cached_info
        self._last_prediction_signature = sig
        if len(prefix) >= 2 and len(prefix) < 10 and np.linalg.norm(prefix[0] - prefix[-1]) < 0.5:
            self.route_belief = {}
        classes, probs, unc = self.model.predict_segment_ml(prefix)
        DECAY = 0.60
        new_belief = {k: DECAY * v for k, v in self.route_belief.items()}
        for cls, p in zip(classes, probs):
            new_belief[cls] = new_belief.get(cls, 0.0) + (1.0 - DECAY) * float(p)
        s = sum(new_belief.values()) + 1e-12
        self.route_belief = {k: v / s for k, v in new_belief.items()}
        route_pair = max(self.route_belief, key=self.route_belief.get)
        route_prob = self.route_belief[route_pair]

        pm = motion_mode_probs(prefix, prefix[-max(6, self.window):])
        self.mode = max(pm, key=pm.get)

        # ---- Topology-aware classification ----
        zone = classify_topology_state(self.model, prefix[-1])["zone"]
        base_conf = 0.5 * route_prob + 0.5 * max(pm.values())
        if zone == "I":
            display_label = "INTERSECTION"
        elif zone is None:
            display_label = f"ROUTE {route_pair[0]}→{route_pair[1]}"
        else:
            # Motion direction relative to the intersection center.
            I_center = self.model.intersection_poly.mean(axis=0)
            current_dist = np.linalg.norm(prefix[-1] - I_center)
            prev_dist = np.linalg.norm(prefix[-5] - I_center) if len(prefix) >= 5 else current_dist
            direction = "→I" if current_dist < prev_dist else "I→"
            display_label = f"B{zone} {direction}"
        topology_confidence = max(pm.values()) if zone == "I" else route_prob
        self._cached_info = {
            "topology_zone": zone,
            "topology_label": display_label,
            "route_pair": route_pair,
            "route_prob": route_prob,
            "route_uncertainty": unc,
            "mode": self.mode,
            "mode_probs": pm,
            "display_label": display_label,
            "display_confidence": base_conf,
            "topology_confidence": topology_confidence,
        }
        return self._cached_info

# ==========================================================
# Forecasting (progress-aware optional)
# ==========================================================
def blend_velocity(mu_local, cov_local, mu_class, cov_class, seg_len, kappa=0.002):
    alpha = 1.0 - np.exp(-kappa * max(1, seg_len))
    return (alpha * mu_local + (1 - alpha) * mu_class,
            alpha * cov_local + (1 - alpha) * cov_class,
            alpha)

def class_progress_velocity(model, cid, est_progress):
    bins = model.progress_bins
    b = int(np.clip(np.round((len(bins) - 1) * est_progress), 0, len(bins) - 1))
    st = model.class_progress_stats.get(cid)
    if not st:
        return model.class_step_stats.get(cid, (np.zeros(2), np.eye(2) * 1e-3))
    return st["mus"][b], st["covs"][b]

def estimate_progress_for_class(prefix, model, cid):
    d = np.linalg.norm(np.diff(prefix, axis=0), axis=1)
    obs_len = float(d.sum()) if len(d) else 0.0
    avg_len = model.class_progress_stats.get(cid, {}).get("avg_len", 1.0)
    return float(np.clip(obs_len / max(1e-6, avg_len), 0.0, 1.0))

def forecast_particles(prefix, online, horizon, n_particles=1500, seed=0, use_progress_dyn=False,
                       loading_drift_scale=0.4, loading_cov_scale=1.2, loading_anchor_alpha=0.7,
                       stationary_cov_scale=0.25, stationary_eps=5e-5, cov_floor=1e-6):
    """Forecast particle endpoints `horizon` steps ahead, conditioned on the
    micro-mode, local kinematics and class priors (optionally progress-aware)."""
    rng = np.random.default_rng(seed)
    # Local kinematics from recent window.
    seg = prefix[-max(6, online.window):]
    dif = np.diff(seg, axis=0)
    if dif.shape[0] == 0:
        mu_local, cov_local = np.zeros(2), np.eye(2) * 1e-3
    elif dif.shape[0] == 1:
        mu_local, cov_local = dif[0], np.eye(2) * 1e-3
    else:
        mu_local, cov_local = dif.mean(axis=0), np.cov(dif.T)
    cov_local = cov_local + 1e-12 * np.eye(2)
    # Class stats.
    info = online._cached_info if online._cached_info is not None else online.update(prefix)
    route_pair = info.get("route_pair")
    if route_pair not in online.model.pair_to_cid:
        route_pair = online.model.predict_segment_ml(prefix, topk=1)[0][0]
    cid = online.model.pair_to_cid[route_pair]
    mu_c, cov_c = online.model.class_step_stats.get(cid, (np.zeros(2), np.eye(2) * 1e-3))
    if use_progress_dyn:
        try:
            mu_c, cov_c = class_progress_velocity(
                online.model, cid, estimate_progress_for_class(prefix, online.model, cid))
        except Exception:
            mu_c, cov_c = online.model.class_step_stats.get(cid, (np.zeros(2), np.eye(2) * 1e-3))
    # Select dynamics by micro-mode.
    if info["mode"] == "moving":
        mu, cov, _ = blend_velocity(mu_local, cov_local, mu_c, cov_c, len(prefix))
        x0 = prefix[-1]
    elif info["mode"] == "loading":
        # Slow drift + jitter; anchor near centroid, keep some last-point weight.
        mu_load, cov_load = online.model.__dict__.get("class_loading_stats", {}).get(cid, (mu_c, cov_c))
        mu_blend, cov_blend, _ = blend_velocity(mu_local, cov_local, mu_load, cov_load, len(prefix))
        mu = loading_drift_scale * mu_blend
        cov = loading_cov_scale * cov_blend + 1e-4 * np.eye(2)
        x0 = loading_anchor_alpha * seg.mean(axis=0) + (1.0 - loading_anchor_alpha) * prefix[-1]
    else:  # stationary / parked
        x0 = seg.mean(axis=0)
        mu = np.zeros(2)
        cov = stationary_cov_scale * cov_local + stationary_eps * np.eye(2)
    cov = cov + cov_floor * np.eye(2)
    # Monte Carlo rollout (endpoints only).
    parts = np.tile(x0, (n_particles, 1))
    if horizon > 0:
        parts += rng.multivariate_normal(mean=mu, cov=cov, size=(horizon, n_particles)).sum(axis=0)
    return info, parts

# ---- Heatmap + CI95 drawing ----
def _ci95_ellipse(parts):
    """(mu, ellipse_points) at the 95% level (chi2(2) = 5.991)."""
    mu = parts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(parts.T))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = 5.991
    t = np.linspace(0, 2 * np.pi, 300)
    return mu, np.c_[np.sqrt(k * vals[0]) * np.cos(t), np.sqrt(k * vals[1]) * np.sin(t)] @ vecs.T + mu

def draw_heatmap_and_ci95_on_ax(ax, parts, label="CI95", cmap="magma"):
    H2, xe, ye = np.histogram2d(parts[:, 0], parts[:, 1], bins=80)
    ax.pcolormesh(xe, ye, H2.T, cmap=cmap, shading="auto", alpha=0.70, edgecolors="none")
    _mu, ell = _ci95_ellipse(parts)
    ax.plot(ell[:, 0], ell[:, 1], color="cyan", lw=2, label=label)
    ax.autoscale(enable=True, axis="both", tight=True)
    ax.set_aspect("equal", "box")

def plot_future_heatmap_with_CI95(prefix, online, horizon=1, n_particles=2000, seed=0,
                                  use_progress_dyn=False, ax=None, title_prefix="H="):
    info, parts = forecast_particles(prefix, online, horizon=horizon, n_particles=n_particles,
                                     seed=seed, use_progress_dyn=use_progress_dyn)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    H2, xe, ye = np.histogram2d(parts[:, 0], parts[:, 1], bins=120)
    ax.pcolormesh(xe, ye, H2.T, cmap="magma", shading="auto")
    _mu, ell = _ci95_ellipse(parts)
    ax.plot(ell[:, 0], ell[:, 1], color="cyan", lw=2, label="CI95 ellipse")
    x0 = prefix[-1]
    ax.scatter([x0[0]], [x0[1]], c="yellow", edgecolor="k", s=70, zorder=5, label="Current")
    ax.set_title(f"{title_prefix}{horizon} — {info['display_label']} (conf={info['display_confidence']:.2f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax

# ==========================================================
# Online pool (per-track state, slider-safe replay)
# ==========================================================
class OnlinePool:
    """One OnlineClassifier per track_id; incremental O(delta) replay so the
    slider can move both directions without reprocessing history."""

    def __init__(self, model, T, window=25, enter_baf=3, exit_baf=5, rev_angle_deg=120.0,
                 baf_sensitivity=1.0, stationary_sens=1.0, bbox_sens=1.0):
        self.model = model
        self.T = T
        self.cfg = dict(window=window, enter_baf=enter_baf, exit_baf=exit_baf,
                        rev_angle_deg=rev_angle_deg, baf_sensitivity=baf_sensitivity,
                        stationary_sens=stationary_sens, bbox_sens=bbox_sens)
        self.store: Dict[int, dict] = {}

    def _new_online(self):
        return OnlineClassifier(self.model, self.T, **self.cfg)

    def update_to_frame(self, tid, tid_hist, target_frame):
        if not tid_hist:
            return None
        if tid not in self.store:
            self.store[tid] = {"online": self._new_online(), "last_frame": -1,
                               "history": deque(), "cursor": 0}
        state = self.store[tid]
        if state["last_frame"] > target_frame:          # slider moved backward
            self.store[tid] = state = {"online": self._new_online(), "last_frame": -1,
                                       "history": deque(), "cursor": 0}
        cursor = state["cursor"]
        while cursor < len(tid_hist) and tid_hist[cursor][0] <= target_frame:
            state["history"].append(tid_hist[cursor][1])
            cursor += 1
        state["cursor"] = cursor
        state["last_frame"] = target_frame
        if len(state["history"]) < 2:
            return state["online"], None
        prefix = np.asarray(state["history"], dtype=float)
        if not EXPORT_FAST_MODE and len(prefix) > 16:
            prefix = smooth_trajectory(prefix, grid=TRAJ_GRID)
        online = state["online"]
        if not hasattr(online, "_last_len") or online._last_len != len(prefix):
            online.update(prefix)
            online._last_len = len(prefix)
        return online, prefix

    def reset(self):
        self.store.clear()

# ==========================================================
# Prediction export record (shared by GUI export and CLI export)
# ==========================================================
def build_prediction_record(tid, info):
    a, b = info["route_pair"]
    return {
        "track_id": int(tid),
        "route": {"from": int(a), "to": int(b),
                  "prob": float(info["route_prob"]),
                  "uncertainty": float(info["route_uncertainty"])},
        "topology": {"zone": info.get("topology_zone"),
                     "label": info.get("topology_label"),
                     "confidence": float(info.get("topology_confidence", 0.0))},
        "mode": info["mode"],
        "mode_probs": {k: float(v) for k, v in info["mode_probs"].items()},
        "display_label": info["display_label"],
        "display_confidence": float(info["display_confidence"]),
    }

# ==========================================================
# GUI (PyQt6)
# ==========================================================
def run_gui(trajs, model, online, K_ENTRIES, K_CODE, N_PCA, WIN, use_progress_dyn,
            frame_list=None, frame_objects=None, tid_frame_history=None,
            orig_frames=None, source_json_path=None):
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from PyQt6 import QtCore, QtWidgets
    except Exception:
        print("PyQt6 is required: pip install PyQt6 matplotlib")
        raise

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    class StartDialog(QtWidgets.QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Load or Train Model")
            v = QtWidgets.QVBoxLayout(self)
            self.list = QtWidgets.QListWidget()
            v.addWidget(QtWidgets.QLabel("Saved models (newest first):"))
            v.addWidget(self.list)
            h = QtWidgets.QHBoxLayout()
            self.btn_load = QtWidgets.QPushButton("Load selected")
            self.btn_train = QtWidgets.QPushButton("Train new")
            self.btn_cancel = QtWidgets.QPushButton("Cancel")
            h.addWidget(self.btn_load); h.addWidget(self.btn_train)
            h.addStretch(1); h.addWidget(self.btn_cancel)
            v.addLayout(h)
            self.btn_cancel.clicked.connect(self.reject)
            self.btn_load.clicked.connect(self.accept_load)
            self.btn_train.clicked.connect(self.accept_train)
            self.files = list_saved_models()
            for p in self.files:
                ts = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
                self.list.addItem(f"{os.path.basename(p)} ({ts})")
            self.choice = None

        def accept_load(self):
            row = self.list.currentRow()
            if row < 0:
                QtWidgets.QMessageBox.warning(self, "No selection", "Select a model.")
                return
            self.choice = ("load", self.files[row])
            self.accept()

        def accept_train(self):
            self.choice = ("train", None)
            self.accept()

    def _make_online(m):
        T = auto_tune_mode_thresholds(trajs, win=WIN, rev_angle_deg=120.0)
        return OnlineClassifier(m, T, window=WIN, enter_baf=3, exit_baf=5, rev_angle_deg=120.0,
                                baf_sensitivity=1.0, stationary_sens=1.0, bbox_sens=1.0)

    if model is None or online is None:
        dlg = StartDialog()
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        ch, path = dlg.choice
        if ch == "load":
            model = ensure_model_topology(MLModel.load(path), trajs)
            online = _make_online(model)
        else:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                model = auto_tune_and_train_ml(trajs, K_ENTRIES, K_CODE, N_PCA)
                savep = timestamped_model_path()
                model.save(savep)
                QtWidgets.QMessageBox.information(None, "Model saved", f"Saved:\n{savep}")
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            online = _make_online(model)

    class TrajPlayer(QtWidgets.QMainWindow):
        """Trajectory monitor: trajectory-mode (single scrub) and frame-mode
        (time-synced multi-object with selection, table and JSON export)."""

        def __init__(self, trajs, model, online, bg_image_path=None, bg_extent=None,
                     bg_origin="lower", bg_alpha=0.85, frame_list=None, frame_objects=None,
                     tid_frame_history=None, orig_frames=None, source_json_path=None):
            super().__init__()
            self.setWindowTitle("Trajectory Monitor — Warehouse forklift")
            self.trajs = trajs
            self.model = model
            self.online_single = online
            self.use_progress_dyn = use_progress_dyn
            self.tids = sorted(trajs.keys()) if trajs else []
            self.cur_tid = self.tids[0] if self.tids else None
            self.cur_idx = 6
            self.frame_list = frame_list
            self.frame_objects = frame_objects
            self.tid_frame_history = tid_frame_history
            self.orig_frames = orig_frames                # kept for API compat (unused)
            self.source_json_path = source_json_path
            self.use_frame_mode = frame_list is not None and frame_objects is not None
            self.current_frame = 0
            self.selected_tid = None
            os_ = self.online_single
            T = auto_tune_mode_thresholds(self.trajs, win=os_.window, rev_angle_deg=os_.rev_angle_deg)
            self.pool = OnlinePool(self.model, T, window=os_.window, enter_baf=os_.enter_baf,
                                   exit_baf=os_.exit_baf, rev_angle_deg=os_.rev_angle_deg,
                                   baf_sensitivity=os_.baf_sensitivity,
                                   stationary_sens=os_.stationary_sens, bbox_sens=os_.bbox_sens)
            self.bg_image_path, self.bg_extent, self.bg_origin, self.bg_alpha = (
                bg_image_path, bg_extent, bg_origin, bg_alpha)
            self._bg_img = None
            self._view_extent = None
            self._was_playing = False
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(80)
            self.timer.timeout.connect(self.step_forward)

            # ---- layout ----
            main = QtWidgets.QWidget(self)
            self.setCentralWidget(main)
            root = QtWidgets.QHBoxLayout(main)
            left = QtWidgets.QVBoxLayout()
            self.fig_main = Figure(figsize=(4, 6), tight_layout=True)
            self.ax_main = self.fig_main.add_subplot(111)
            self.ax_main.set_aspect("equal", adjustable="datalim")
            try:
                self.ax_main.set_box_aspect(1)
            except Exception:
                pass
            self.canvas_main = FigureCanvas(self.fig_main)
            left.addWidget(self.canvas_main, stretch=6)
            ctrl = QtWidgets.QHBoxLayout()
            self.cb = QtWidgets.QComboBox()
            for t in self.tids:
                self.cb.addItem(str(t))
            self.cb.currentIndexChanged.connect(self.change_tid)
            self.btn_play = QtWidgets.QPushButton("Play")
            self.btn_play.clicked.connect(self.toggle)
            self.btn_b = QtWidgets.QPushButton("◀ Step")
            self.btn_b.clicked.connect(self.step_back)
            self.btn_f = QtWidgets.QPushButton("Step ▶")
            self.btn_f.clicked.connect(self.step_forward)
            self.btn_r = QtWidgets.QPushButton("Reset")
            self.btn_r.clicked.connect(self.reset_pos)
            ctrl.addWidget(QtWidgets.QLabel("Trajectory:"))
            ctrl.addWidget(self.cb)
            for b in (self.btn_play, self.btn_b, self.btn_f, self.btn_r):
                ctrl.addWidget(b)
            ctrl.addStretch(1)
            left.addLayout(ctrl)
            ctrl3 = QtWidgets.QHBoxLayout()
            self.cb_objects = QtWidgets.QComboBox()
            self.cb_objects.currentIndexChanged.connect(self.on_object_selected)
            self.btn_save_pred = QtWidgets.QPushButton("Save predictions JSON")
            self.btn_save_pred.clicked.connect(self.save_predictions_dialog)
            if self.use_frame_mode:
                ctrl3.addWidget(QtWidgets.QLabel("Object:"))
                ctrl3.addWidget(self.cb_objects)
                ctrl3.addStretch(1)
                ctrl3.addWidget(self.btn_save_pred)
                left.addLayout(ctrl3)
            ctrl2 = QtWidgets.QHBoxLayout()
            ctrl2.addWidget(QtWidgets.QLabel("Position"))
            self.sld_prefix = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.lbl_prefix = QtWidgets.QLabel("")
            self.sld_prefix.valueChanged.connect(self._on_prefix_slider_changed)
            self.sld_prefix.sliderPressed.connect(self._on_prefix_slider_pressed)
            self.sld_prefix.sliderReleased.connect(self._on_prefix_slider_released)
            ctrl2.addWidget(self.sld_prefix, stretch=1)
            ctrl2.addWidget(self.lbl_prefix)
            left.addLayout(ctrl2)
            right = QtWidgets.QVBoxLayout()
            self.fig_route = Figure(figsize=(4, 3), tight_layout=True)
            self.ax_route = self.fig_route.add_subplot(111)
            self.canvas_route = FigureCanvas(self.fig_route)
            right.addWidget(self.canvas_route, stretch=1)
            self.fig_micro = Figure(figsize=(4, 3), tight_layout=True)
            self.ax_micro = self.fig_micro.add_subplot(111)
            self.canvas_micro = FigureCanvas(self.fig_micro)
            right.addWidget(self.canvas_micro, stretch=1)
            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels(
                ["track_id", "label", "conf", "mode", "p(mov)", "p(load)", "p(stat)"])
            self.table.horizontalHeader().setStretchLastSection(True)
            if self.use_frame_mode:
                right.addWidget(self.table, stretch=1)
            # Tuning panel (NOTE: sensitivity values are stored but not applied yet)
            pane = QtWidgets.QGroupBox("Maneuvers Tuning")
            form = QtWidgets.QFormLayout(pane)
            self.sld_baf = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.sld_baf.setRange(5, 30)
            self.sld_baf.setValue(int(os_.baf_sensitivity * 10))
            self.lbl_baf = QtWidgets.QLabel(f"{os_.baf_sensitivity:.2f}")
            self.sld_baf.valueChanged.connect(lambda _: self._update_sens())
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.sld_baf); row.addWidget(self.lbl_baf)
            form.addRow("Loading sensitivity (0.5-3.0)", row)
            self.spn_rev = QtWidgets.QSpinBox()
            self.spn_rev.setRange(60, 180)
            self.spn_rev.setValue(int(os_.rev_angle_deg))
            self.spn_rev.valueChanged.connect(self._apply_tuning)
            form.addRow("Reversal angle (deg)", self.spn_rev)
            self.spn_win = QtWidgets.QSpinBox()
            self.spn_win.setRange(5, 60)
            self.spn_win.setValue(os_.window)
            self.spn_win.valueChanged.connect(self._apply_tuning)
            form.addRow("Window size", self.spn_win)
            self.spn_enter = QtWidgets.QSpinBox()
            self.spn_enter.setRange(1, 20)
            self.spn_enter.setValue(os_.enter_baf)
            self.spn_enter.valueChanged.connect(self._apply_tuning)
            form.addRow("Loading enter frames", self.spn_enter)
            self.spn_exit = QtWidgets.QSpinBox()
            self.spn_exit.setRange(1, 20)
            self.spn_exit.setValue(os_.exit_baf)
            self.spn_exit.valueChanged.connect(self._apply_tuning)
            form.addRow("Loading exit frames", self.spn_exit)
            self.sld_stat = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.sld_stat.setRange(5, 30)
            self.sld_stat.setValue(int(os_.stationary_sens * 10))
            self.lbl_stat = QtWidgets.QLabel(f"{os_.stationary_sens:.2f}")
            self.sld_stat.valueChanged.connect(lambda _: self._update_sens())
            row2 = QtWidgets.QHBoxLayout()
            row2.addWidget(self.sld_stat); row2.addWidget(self.lbl_stat)
            form.addRow("Stationary sensitivity (0.5-3.0)", row2)
            self.sld_bbox = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.sld_bbox.setRange(5, 30)
            self.sld_bbox.setValue(int(os_.bbox_sens * 10))
            self.lbl_bbox = QtWidgets.QLabel(f"{os_.bbox_sens:.2f}")
            self.sld_bbox.valueChanged.connect(lambda _: self._update_sens())
            row3 = QtWidgets.QHBoxLayout()
            row3.addWidget(self.sld_bbox); row3.addWidget(self.lbl_bbox)
            form.addRow("BBox tightness sensitivity (0.5-3.0)", row3)
            self.chk_overlay = QtWidgets.QCheckBox("Show forecasted future positions (5 frames ahead)")
            self.chk_overlay.setChecked(True)
            self.chk_overlay.stateChanged.connect(self.update_all)
            form.addRow(self.chk_overlay)
            self.btn_heat = QtWidgets.QPushButton("Show heatmaps (H=1,2,3)")
            self.btn_heat.clicked.connect(self.show_heatmaps)
            form.addRow(self.btn_heat)
            right.addWidget(pane, stretch=0)
            root.addLayout(left, stretch=3)
            root.addLayout(right, stretch=2)
            self.current_bev = None
            self.current_bev_ts = None
            self._prepare_background_config()
            self._setup_prefix_slider()
            self.update_all()

        def on_object_selected(self):
            try:
                self.selected_tid = int(self.cb_objects.currentText())
            except Exception:
                pass
            self.update_all()

        # ---------- background ----------
        def load_bev_image(self, timestamp):
            if not bev_lookup:
                return None
            ts = int(timestamp)
            if ts not in bev_lookup:
                keys = np.asarray(list(bev_lookup.keys()), dtype=np.int64)
                ts = int(keys[np.argmin(np.abs(keys - ts))])
            path = os.path.join(BEV_FOLDER, bev_lookup[ts])
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

        def vis_xy(self, x, y):
            if self.current_bev is None:
                return x, y
            return world_to_occupancy(x, y)

        def vis_poly(self, poly):
            poly = np.asarray(poly)
            return np.asarray([self.vis_xy(x, y) for x, y in poly], dtype=float)

        def _data_bounds(self):
            if self.use_frame_mode and self.frame_objects:
                all_pts = [xy for fr in self.frame_objects for xy in fr.values()]
                if not all_pts:
                    all_pts = [np.zeros(2)]
                all_pts = np.asarray(all_pts, dtype=float)
            else:
                all_pts = np.vstack(list(self.trajs.values())) if self.trajs else np.zeros((1, 2))
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            pad_x = 0.05 * (xmax - xmin + 1e-9)
            pad_y = 0.05 * (ymax - ymin + 1e-9)
            return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y

        def _square_from_bounds(self, xmin, xmax, ymin, ymax):
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            span = max(xmax - xmin, ymax - ymin)
            return (cx - 0.5 * span, cx + 0.5 * span, cy - 0.5 * span, cy + 0.5 * span)

        def _prepare_background_config(self):
            if self.bg_extent is None:
                self._view_extent = self._square_from_bounds(*self._data_bounds())
            else:
                self._view_extent = self._square_from_bounds(*self.bg_extent)
            if self.bg_image_path:
                try:
                    self._bg_img = mpimg.imread(self.bg_image_path)
                except Exception as e:
                    print(f"[TrajPlayer] Could not load background image: {e}")
                    self._bg_img = None

        # ---------- slider ----------
        def _setup_prefix_slider(self):
            if self.use_frame_mode:
                minv, maxv = 0, max(0, len(self.frame_list) - 1)
                self.sld_prefix.setValue(self.current_frame)
            else:
                arr = self.trajs[self.cur_tid] if self.cur_tid is not None else np.zeros((7, 2))
                minv, maxv = 6, max(7, len(arr) - 1)
                self.sld_prefix.setValue(max(minv, min(self.cur_idx, maxv)))
            self.sld_prefix.blockSignals(True)
            self.sld_prefix.setRange(minv, maxv)
            page = max(1, (maxv - minv) // 10)
            self.sld_prefix.setPageStep(page)
            self.sld_prefix.setSingleStep(1)
            self.sld_prefix.setTickInterval(page)
            self.sld_prefix.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            if self.use_frame_mode:
                self.sld_prefix.setValue(self.current_frame)
            else:
                self.sld_prefix.setValue(max(minv, min(self.cur_idx, maxv)))
            self.sld_prefix.blockSignals(False)
            self._update_prefix_label()

        def _update_prefix_label(self):
            if self.use_frame_mode:
                ts = self.frame_list[self.current_frame] if self.frame_list else None
                self.lbl_prefix.setText(f"frame {self.current_frame + 1}/{len(self.frame_list)}  ts={ts}")
            else:
                total = len(self.trajs[self.cur_tid]) if self.cur_tid is not None else 0
                self.lbl_prefix.setText(f"{self.cur_idx}/{total}")

        def _sync_prefix_slider(self):
            self.sld_prefix.blockSignals(True)
            self.sld_prefix.setValue(self.current_frame if self.use_frame_mode else self.cur_idx)
            self.sld_prefix.blockSignals(False)
            self._update_prefix_label()

        def _on_prefix_slider_pressed(self):
            self._was_playing = self.timer.isActive()
            if self._was_playing:
                self.timer.stop()

        def _on_prefix_slider_released(self):
            if self._was_playing:
                self.timer.start()
            self._was_playing = False

        def _on_prefix_slider_changed(self, val):
            if self.use_frame_mode:
                self.current_frame = val
            else:
                arr = self.trajs[self.cur_tid]
                self.cur_idx = max(6, min(val, len(arr) - 1))
            self._update_prefix_label()
            self.update_all()

        # ---------- tuning ----------
        def _update_sens(self):
            s_baf = self.sld_baf.value() / 10.0
            s_stat = self.sld_stat.value() / 10.0
            s_bbox = self.sld_bbox.value() / 10.0
            self.online_single.baf_sensitivity = s_baf
            self.lbl_baf.setText(f"{s_baf:.2f}")
            self.online_single.stationary_sens = s_stat
            self.lbl_stat.setText(f"{s_stat:.2f}")
            self.online_single.bbox_sens = s_bbox
            self.lbl_bbox.setText(f"{s_bbox:.2f}")
            self.pool.cfg.update(baf_sensitivity=s_baf, stationary_sens=s_stat, bbox_sens=s_bbox)
            self.pool.reset()
            self.update_all()

        def _apply_tuning(self):
            self.online_single.rev_angle_deg = float(self.spn_rev.value())
            self.online_single.window = int(self.spn_win.value())
            self.online_single.enter_baf = int(self.spn_enter.value())
            self.online_single.exit_baf = int(self.spn_exit.value())
            self.pool.cfg.update(window=self.online_single.window,
                                 enter_baf=self.online_single.enter_baf,
                                 exit_baf=self.online_single.exit_baf,
                                 rev_angle_deg=self.online_single.rev_angle_deg)
            self.pool.reset()
            self.update_all()

        # ---------- playback ----------
        def change_tid(self, idx):
            if not self.tids:
                return
            self.cur_tid = self.tids[idx]
            self.cur_idx = 6
            self._setup_prefix_slider()
            self.update_all()

        def toggle(self):
            if self.timer.isActive():
                self.timer.stop()
            else:
                self.timer.start()
            self.btn_play.setText("Pause" if self.timer.isActive() else "Play")

        def step_forward(self):
            if self.use_frame_mode:
                self.current_frame = min(len(self.frame_list) - 1, self.current_frame + 1)
                self.update_all()
                if self.current_frame >= len(self.frame_list) - 1:
                    self.timer.stop()
                    self.btn_play.setText("Play")
            else:
                arr = self.trajs[self.cur_tid]
                self.cur_idx = min(len(arr) - 1, self.cur_idx + 1)
                self.update_all()
                if self.cur_idx >= len(arr) - 1:
                    self.timer.stop()
                    self.btn_play.setText("Play")

        def step_back(self):
            if self.use_frame_mode:
                self.current_frame = max(0, self.current_frame - 1)
            else:
                self.cur_idx = max(6, self.cur_idx - 1)
            self.update_all()

        def reset_pos(self):
            if self.use_frame_mode:
                self.current_frame = 0
                self.pool.reset()
            else:
                self.cur_idx = 6
            self.update_all()

        # ---------- drawing ----------
        def _draw_background(self):
            if self.current_bev is not None:
                h, w = self.current_bev.shape[:2]
                self.ax_main.imshow(self.current_bev, extent=(0, w, h, 0), zorder=0)
                self.ax_main.set_xlim(0, w)
                self.ax_main.set_ylim(h, 0)
                self.ax_main.autoscale(False)
            elif self._bg_img is not None and self._view_extent is not None:
                self.ax_main.imshow(self._bg_img, extent=self._view_extent,
                                    origin=self.bg_origin, alpha=self.bg_alpha, zorder=0)
            self.ax_main.set_aspect("equal")

        def draw_topology(self):
            if self.model.zone_polys:
                for zid, poly in sorted(self.model.zone_polys.items()):
                    poly_draw = self.vis_poly(poly)
                    q = np.vstack([poly_draw, poly_draw[0]])
                    self.ax_main.plot(q[:, 0], q[:, 1], color="deepskyblue", lw=1.0,
                                      ls="--", alpha=0.55, zorder=1)
                    c = poly_draw.mean(axis=0)
                    self.ax_main.text(c[0], c[1], f"B{zid}", color="deepskyblue", fontsize=8,
                                      ha="center", va="center", zorder=2)
            if self.model.intersection_poly is not None and len(self.model.intersection_poly) >= 3:
                poly_draw = self.vis_poly(self.model.intersection_poly)
                q = np.vstack([poly_draw, poly_draw[0]])
                self.ax_main.fill(poly_draw[:, 0], poly_draw[:, 1], color="gold",
                                  alpha=0.15, zorder=1)
                self.ax_main.plot(q[:, 0], q[:, 1], color="gold", lw=2.5, zorder=2)
                c = poly_draw.mean(axis=0)
                self.ax_main.text(c[0], c[1], "INTERSECTION", color="darkorange", weight="bold",
                                  fontsize=10, ha="center", va="center",
                                  bbox=dict(boxstyle="round,pad=0.25", fc="black",
                                            ec="gold", alpha=0.7),
                                  zorder=7)

        def draw_entries(self):
            self.draw_topology()
            colors = plt.cm.tab10.colors
            for cid, (center, w, h, ang) in enumerate(self.model.entry_ellipses):
                E_occ = self.vis_poly(ellipse_poly(center, w, h, ang))
                cx, cy = self.vis_xy(center[0], center[1])
                self.ax_main.plot(E_occ[:, 0], E_occ[:, 1], color=colors[cid % 10],
                                  lw=2, alpha=0.9, zorder=2)
                self.ax_main.scatter([cx], [cy], c=[colors[cid % 10]], s=60,
                                     edgecolor="k", zorder=5)
                self.ax_main.text(cx, cy, f"ID {cid}", fontsize=9, ha="center", va="bottom")

        # ---------- main update ----------
        def update_all(self):
            if self.use_frame_mode and self.frame_list:
                ts = self.frame_list[self.current_frame]
                if ts != self.current_bev_ts:
                    self.current_bev = self.load_bev_image(ts)
                    self.current_bev_ts = ts
            self.ax_main.clear()
            self._draw_background()
            if self.use_frame_mode:
                objs = self.frame_objects[self.current_frame]
                active_tids = sorted(objs.keys())
                # Preserve current selection across refreshes.
                old_text = self.cb_objects.currentText()
                self.cb_objects.blockSignals(True)
                self.cb_objects.clear()
                for tid in active_tids:
                    self.cb_objects.addItem(str(tid))
                if old_text and self.cb_objects.findText(old_text) >= 0:
                    self.cb_objects.setCurrentIndex(self.cb_objects.findText(old_text))
                self.cb_objects.blockSignals(False)
                for tid, (x, y) in objs.items():
                    px, py = self.vis_xy(x, y)
                    self.ax_main.scatter([px], [py], s=70, label=f"ID {tid}", zorder=4)
                    self.ax_main.text(px, py, str(tid), fontsize=9, zorder=5)
                self.draw_entries()
                if not active_tids:
                    self.ax_main.set_title(
                        f"Frame {self.current_frame + 1}/{len(self.frame_list)} — no active objects")
                    self.canvas_main.draw()
                    self.ax_route.clear(); self.canvas_route.draw()
                    self.ax_micro.clear(); self.canvas_micro.draw()
                    self.table.setRowCount(0)
                    self._sync_prefix_slider()
                    return
                sel_tid = self.selected_tid if self.selected_tid in active_tids else active_tids[0]
                self.selected_tid = sel_tid
                online_sel, prefix_sel = self.pool.update_to_frame(
                    sel_tid, self.tid_frame_history.get(sel_tid, []), self.current_frame)
                if prefix_sel is not None:
                    prefix_draw = np.asarray([self.vis_xy(x, y) for x, y in prefix_sel])
                    self.ax_main.plot(prefix_draw[:, 0], prefix_draw[:, 1],
                                      color="tab:blue", lw=2, zorder=3)
                    self.ax_main.scatter(prefix_draw[-1, 0], prefix_draw[-1, 1], c="yellow",
                                         edgecolor="k", s=60, zorder=6)
                if prefix_sel is not None and len(prefix_sel) >= 6:
                    classes, probs, unc = self.model.predict_segment_ml(prefix_sel)
                    info = online_sel._cached_info
                    labels = [f"{a}->{b}" for (a, b) in classes]
                    self.ax_route.clear()
                    self.ax_route.barh(labels[::-1], probs[::-1], color="tab:green")
                    self.ax_route.set_xlim(0, 1)
                    self.ax_route.set_title(f"Route (unc={unc:.2f})")
                    self.canvas_route.draw()
                    mp = info["mode_probs"]
                    self.ax_micro.clear()
                    self.ax_micro.barh(["moving", "loading", "stationary"],
                                       [mp["moving"], mp["loading"], mp["stationary"]],
                                       color=["tab:gray", "tab:orange", "tab:green"])
                    self.ax_micro.set_xlim(0, 1)
                    self.ax_micro.set_title(f"Maneuver = {info['mode']}")
                    self.canvas_micro.draw()
                    if self.chk_overlay.isChecked():
                        _, parts = forecast_particles(prefix_sel, online_sel, horizon=5,
                                                      n_particles=1200, seed=42,
                                                      use_progress_dyn=self.use_progress_dyn)
                        parts_vis = np.asarray([self.vis_xy(x, y) for x, y in parts])
                        draw_heatmap_and_ci95_on_ax(self.ax_main, parts_vis, label="CI95 (H=1)")
                    self.ax_main.set_title(
                        f"Frame {self.current_frame + 1}/{len(self.frame_list)} — Selected TID {sel_tid} "
                        f"{info['display_label']} (conf={info['display_confidence']:.2f})")
                else:
                    self.ax_route.clear()
                    self.ax_route.set_title("Route (insufficient prefix)")
                    self.canvas_route.draw()
                    self.ax_micro.clear()
                    self.ax_micro.set_title("Maneuver (insufficient prefix)")
                    self.canvas_micro.draw()
                    self.ax_main.set_title(
                        f"Frame {self.current_frame + 1}/{len(self.frame_list)} — "
                        f"Selected TID {sel_tid} (insufficient prefix)")
                # Table: predictions for all active objects.
                rows = []
                for tid in active_tids:
                    online_i, prefix_i = self.pool.update_to_frame(
                        tid, self.tid_frame_history.get(tid, []), self.current_frame)
                    if prefix_i is None or len(prefix_i) < 6:
                        rows.append((tid, "-", np.nan, "-", np.nan, np.nan, np.nan))
                        continue
                    info_i = online_i._cached_info
                    rows.append((tid, info_i["display_label"], info_i["display_confidence"],
                                 info_i["mode"], info_i["mode_probs"]["moving"],
                                 info_i["mode_probs"]["loading"],
                                 info_i["mode_probs"]["stationary"]))
                self.table.setRowCount(len(rows))
                for r, row in enumerate(rows):
                    for c, val in enumerate(row):
                        self.table.setItem(r, c, QtWidgets.QTableWidgetItem(
                            "" if (isinstance(val, float) and np.isnan(val)) else str(val)))
                self.ax_main.set_xlabel("X")
                self.ax_main.set_ylabel("Y")
                self.ax_main.legend(loc="best")
                self.canvas_main.draw()
                self._sync_prefix_slider()
                return
            # ---- trajectory mode ----
            if not self.tids:
                self.ax_main.set_title("No trajectories loaded")
                self.canvas_main.draw()
                return
            arr = self.trajs[self.cur_tid]
            self.cur_idx = max(6, self.cur_idx)
            seg = arr[:self.cur_idx]
            arr_draw = np.asarray([self.vis_xy(x, y) for x, y in arr])
            seg_draw = np.asarray([self.vis_xy(x, y) for x, y in seg])
            self.ax_main.plot(arr_draw[:, 0], arr_draw[:, 1], color="lightgray",
                              lw=0.8, alpha=0.6, zorder=1)
            self.ax_main.plot(seg_draw[:, 0], seg_draw[:, 1], color="tab:blue", lw=2, zorder=3)
            self.ax_main.scatter(seg_draw[-1, 0], seg_draw[-1, 1], c="yellow",
                                 edgecolor="k", s=60, zorder=5)
            self.draw_entries()
            info = self.online_single.update(seg)
            if self.chk_overlay.isChecked():
                _, parts = forecast_particles(seg, self.online_single, horizon=5,
                                              n_particles=1200, seed=42,
                                              use_progress_dyn=self.use_progress_dyn)
                parts_vis = np.asarray([self.vis_xy(x, y) for x, y in parts])
                draw_heatmap_and_ci95_on_ax(self.ax_main, parts_vis, label="CI95 (H=1)")
            self.ax_main.set_title(
                f"Trajectory {self.cur_tid} [{self.cur_idx}/{len(arr)}] "
                f"{info['display_label']} (conf={info['display_confidence']:.2f})")
            self.ax_main.set_xlabel("X")
            self.ax_main.set_ylabel("Y")
            self.ax_main.legend(loc="best")
            self.canvas_main.draw()
            classes, probs, unc = self.model.predict_segment_ml(seg)
            labels = [f"{a}->{b}" for (a, b) in classes]
            self.ax_route.clear()
            self.ax_route.barh(labels[::-1], probs[::-1], color="tab:green")
            self.ax_route.set_xlim(0, 1)
            self.ax_route.set_title(f"Route (unc={unc:.2f})")
            self.canvas_route.draw()
            mp = info["mode_probs"]
            self.ax_micro.clear()
            self.ax_micro.barh(["moving", "loading", "stationary"],
                               [mp["moving"], mp["loading"], mp["stationary"]],
                               color=["tab:gray", "tab:orange", "tab:green"])
            self.ax_micro.set_xlim(0, 1)
            self.ax_micro.set_title(f"Maneuver = {info['mode']}")
            self.canvas_micro.draw()
            self._sync_prefix_slider()

        # ---------- heatmaps dialog ----------
        def show_heatmaps(self):
            parts_list = []
            if self.use_frame_mode:
                try:
                    sel_tid = int(self.cb_objects.currentText())
                except Exception:
                    QtWidgets.QMessageBox.information(self, "No object", "No active object selected.")
                    return
                online_sel, prefix_sel = self.pool.update_to_frame(
                    sel_tid, self.tid_frame_history.get(sel_tid, []), self.current_frame)
                if prefix_sel is None or len(prefix_sel) < 6:
                    QtWidgets.QMessageBox.information(
                        self, "Insufficient prefix", "Not enough history to plot heatmaps.")
                    return
                for H in (1, 2, 3):
                    _, p = forecast_particles(prefix_sel, online_sel, horizon=H, n_particles=1500,
                                              seed=H, use_progress_dyn=self.use_progress_dyn)
                    parts_list.append(p)
            else:
                seg = self.trajs[self.cur_tid][:self.cur_idx]
                for H in (1, 2, 3):
                    _, p = forecast_particles(seg, self.online_single, horizon=H, n_particles=1500,
                                              seed=H, use_progress_dyn=self.use_progress_dyn)
                    parts_list.append(p)
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Future position heatmaps (H=1,2,3)")
            lay = QtWidgets.QVBoxLayout(dlg)
            fig = Figure(figsize=(10, 3), tight_layout=True)
            axs = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
            for H, ax, parts in zip([1, 2, 3], axs, parts_list):
                H2, xe, ye = np.histogram2d(parts[:, 0], parts[:, 1], bins=100)
                ax.pcolormesh(xe, ye, H2.T, cmap="magma", shading="auto")
                ax.scatter([parts.mean(axis=0)[0]], [parts.mean(axis=0)[1]], c="cyan", s=25)
                ax.set_title(f"H={H}")
            lay.addWidget(FigureCanvas(fig))
            dlg.exec()

        # ---------- export predictions to JSON ----------
        def _compute_predictions_all_frames(self):
            if not self.use_frame_mode:
                return []
            self.pool.reset()                     # clean deterministic pass
            out = []
            for fi, ts in enumerate(self.frame_list):
                preds = []
                for tid in sorted(self.frame_objects[fi].keys()):
                    online_i, prefix_i = self.pool.update_to_frame(
                        tid, self.tid_frame_history.get(tid, []), fi)
                    if prefix_i is not None and len(prefix_i) >= 6:
                        preds.append(build_prediction_record(tid, online_i._cached_info))
                out.append({"time_stamp": ts, "predictions": preds})
            return out

        def save_predictions_dialog(self):
            if not self.use_frame_mode:
                QtWidgets.QMessageBox.information(
                    self, "Frame-mode only", "Export is only available in frame-mode.")
                return
            preds = self._compute_predictions_all_frames()
            if not preds:
                QtWidgets.QMessageBox.information(self, "No predictions", "No predictions to save.")
                return
            default_name = "predictions_" + (os.path.basename(self.source_json_path)
                                             if self.source_json_path else "frames.json")
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save predictions", default_name, "JSON (*.json)")
            if not out_path:
                return
            try:
                with open(out_path, "w") as f:
                    json.dump(preds, f, indent=2)
                QtWidgets.QMessageBox.information(self, "Saved", f"Predictions saved to:\n{out_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not save predictions:\n{e}")

    w = TrajPlayer(trajs, model, online,
                   frame_list=frame_list, frame_objects=frame_objects,
                   tid_frame_history=tid_frame_history, orig_frames=orig_frames,
                   source_json_path=source_json_path)
    w.resize(1280, 900)
    w.show()
    sys.exit(app.exec())

# ==========================================================
# CLI
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Trajectory ML + Micro Maneuvers + GUI (frame-aware)")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to trajectories JSON")
    parser.add_argument("--entries", type=int, default=7, help="K (entry areas)")
    parser.add_argument("--codebook", type=int, default=16, help="K (step codebook)")
    parser.add_argument("--pca", type=int, default=5, help="PCA components for shape")
    parser.add_argument("--win", type=int, default=25, help="Window size for micro mode")
    parser.add_argument("--progress-dyn", action="store_true", help="Use class progress-aware dynamics")
    parser.add_argument("--demo", action="store_true", help="Train & forecast demo (no GUI)")
    parser.add_argument("--report", action="store_true", help="Create accuracy vs prefix report PNG")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (load or train)")
    parser.add_argument("--export-pred", type=str, default=None,
                        help="Export predictions to JSON (frame-based input only, headless)")
    args = parser.parse_args()

    def select_or_train_model(trajs):
        def train():
            print("\nTraining new model...\n")
            m = auto_tune_and_train_ml(trajs, K_ENTRIES=args.entries,
                                       K_CODE=args.codebook, N_PCA=args.pca)
            p = timestamped_model_path()
            m.save(p)
            print(f"Trained and saved: {p}")
            return m
        saved = sorted(list_saved_models(), key=os.path.getmtime, reverse=True)
        if not saved:
            print("\nNo saved models found.")
            return train()
        if len(saved) == 1:
            print(f"\nLoading saved model:\n  {os.path.basename(saved[0])} "
                  f"({get_model_entry_count(saved[0])} entries)\n")
            return ensure_model_topology(MLModel.load(saved[0]), trajs)
        print("\nMultiple saved models found:\n")
        for idx, path in enumerate(saved, start=1):
            count = get_model_entry_count(path)
            marker = "  <-- matches --entries" if count == args.entries else ""
            print(f"{idx:2d}. {os.path.basename(path)} ({count} entries){marker}")
        print("\n 0. Train new model")
        while True:
            try:
                selection = int(input("\nSelect model number: ").strip())
            except ValueError:
                print("Please enter a valid number.")
                continue
            if selection == 0:
                return train()
            if 1 <= selection <= len(saved):
                chosen = saved[selection - 1]
                print(f"\nLoading model:\n  {os.path.basename(chosen)} "
                      f"({get_model_entry_count(chosen)} entries)\n")
                return ensure_model_topology(MLModel.load(chosen), trajs)
            print("Invalid selection.")

    # ---- detect input format ----
    with open(args.data) as f:
        head = json.load(f)
    is_frame = isinstance(head, list)
    if is_frame:
        frame_list, frame_objects, tid_frame_history, raw_frames = load_frame_based(args.data)
        trajs = {}
        for tid, hist in tid_frame_history.items():
            pts = [xy for (_fi, xy) in hist]
            if len(pts) >= 3:
                trajs[tid] = np.asarray(pts, dtype=float)
        if not trajs:
            print("Warning: No sufficient trajectories extracted "
                  "from frame-based file; training may not work.")
    else:
        frame_list = frame_objects = tid_frame_history = raw_frames = None
        trajs = load_trajectories(args.data)
    print(f"Loaded {'frame-based' if is_frame else 'trajectory-based'} data from {args.data}")

    # ---- report mode ----
    if args.report:
        model = select_or_train_model(trajs)
        res = evaluate_prefix_accuracy(trajs, model)
        out = os.path.join(models_dir(), f"report_acc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        save_accuracy_plot(res, out)
        print(f"Saved accuracy report: {out}")
        return

    # ---- export mode ----
    if is_frame and args.export_pred:
        model = select_or_train_model(trajs)
        T = auto_tune_mode_thresholds(trajs, win=args.win, rev_angle_deg=120.0)
        pool = OnlinePool(model, T, window=args.win, enter_baf=3, exit_baf=5)
        total_frames = len(frame_list)
        print(f"Exporting predictions for {total_frames} frames...")
        pool.reset()
        with open(args.export_pred, "w") as f:
            f.write("[\n")
            first_frame = True
            for fi, ts in enumerate(frame_list):
                if fi % 250 == 0:
                    print(f"\r{fi}/{total_frames} ({100.0 * fi / total_frames:.1f}%)",
                          end="", flush=True)
                preds = []
                for tid in sorted(frame_objects[fi].keys()):
                    online_i, prefix_i = pool.update_to_frame(
                        tid, tid_frame_history.get(tid, []), fi)
                    if prefix_i is not None and len(prefix_i) >= 6:
                        preds.append(build_prediction_record(tid, online_i._cached_info))
                if not first_frame:
                    f.write(",\n")
                json.dump({"time_stamp": ts, "predictions": preds}, f, separators=(",", ":"))
                first_frame = False
            f.write("\n]\n")
        print(f"\nPredictions exported to: {args.export_pred}")
        return

    # ---- demo mode ----
    if args.demo:
        model = auto_tune_and_train_ml(trajs, K_ENTRIES=args.entries,
                                       K_CODE=args.codebook, N_PCA=args.pca)
        savep = timestamped_model_path()
        model.save(savep)
        print(f"Saved model -> {savep}")
        T = auto_tune_mode_thresholds(trajs, win=args.win, rev_angle_deg=120.0)
        online = OnlineClassifier(model, T, window=args.win, enter_baf=3, exit_baf=5)
        if trajs:
            tid = sorted(trajs.keys())[0]
            arr = trajs[tid]
            for frac in [0.15, 0.35, 0.55, 0.75, 0.95]:
                n = max(6, int(len(arr) * frac))
                classes, probs, unc = model.predict_segment_ml(arr[:n])
                print(f"prefix={n:5d} top1={classes[0]} p={probs[0]:.3f} unc={unc:.3f}")
            seg = arr[:min(max(200, len(arr) // 3), len(arr) - 1)]
            for H in (1, 2, 3):
                info, parts = forecast_particles(seg, online, horizon=H, n_particles=3000,
                                                 seed=H, use_progress_dyn=args.progress_dyn)
                H2, xe, ye = np.histogram2d(parts[:, 0], parts[:, 1], bins=100)
                fig, ax = plt.subplots(figsize=(6, 5))
                pcm = ax.pcolormesh(xe, ye, H2.T, cmap="magma", shading="auto")
                ax.scatter([seg[-1, 0]], [seg[-1, 1]], c="yellow", edgecolor="k", s=60, zorder=5)
                ax.set_title(f"{info['display_label']} H={H} conf={info['display_confidence']:.2f} "
                             f"progress_dyn={args.progress_dyn}")
                fig.colorbar(pcm, ax=ax, label="Density")
                out = f"demo_forecast_H{H}.png"
                fig.tight_layout()
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"Saved {out}")
        else:
            print("No trajectories available for demo.")
        return

    # ---- GUI mode ----
    if args.gui:
        model = select_or_train_model(trajs)
        T = auto_tune_mode_thresholds(trajs, win=args.win, rev_angle_deg=120.0)
        online = OnlineClassifier(model, T, window=args.win, enter_baf=3, exit_baf=5)
        run_gui(trajs, model, online, K_ENTRIES=args.entries, K_CODE=args.codebook,
                N_PCA=args.pca, WIN=args.win, use_progress_dyn=True,
                frame_list=frame_list, frame_objects=frame_objects,
                tid_frame_history=tid_frame_history, orig_frames=raw_frames,
                source_json_path=args.data)
        return

    print("Use --demo for a non-GUI run, --report for accuracy plot, "
          "--gui to launch the app, or --export-pred to export predictions (frame-based).")


if __name__ == "__main__":
    main()