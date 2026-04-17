from __future__ import annotations
import os, sys, json, math, argparse
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.image as mpimg

# --------------------- I/O helpers ---------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def models_dir() -> str:
    d = os.path.join(os.getcwd(), "models")
    ensure_dir(d)
    return d

def timestamped_model_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(models_dir(), f"model_{ts}.npz")

def list_saved_models() -> List[str]:
    d = models_dir()
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files

def load_trajectories(json_path: str) -> Dict[int, np.ndarray]:
    """
    Supports two formats:
    (A) trajectory-based: { "12": [[x,y], ...], ... }
    (B) frame-based: [ { "time_stamp": ..., "object_list": [ {track_id, position_3d}, ... ] }, ... ]
    Returns {track_id -> ndarray(N,2)} for training/analytics.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # CASE A: trajectory-based
    if isinstance(raw, dict):
        trajs: Dict[int, np.ndarray] = {}
        for k, v in raw.items():
            try:
                tid = int(k)
            except Exception:
                continue
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
                trajs[tid] = arr
        if not trajs:
            raise RuntimeError("No valid trajectories found in dict-format JSON.")
        return trajs

    # CASE B: frame-based
    if isinstance(raw, list):
        bucket: Dict[int, list] = {}
        for frame in raw:
            objects = frame.get("object_list", [])
            for obj in objects:
                tid = obj.get("track_id", None)
                pos = obj.get("position_3d", None)
                if tid is None or pos is None:
                    continue
                x = pos.get("x", None)
                y = pos.get("y", None)
                if x is None or y is None:
                    continue
                bucket.setdefault(int(tid), []).append([float(x), float(y)])
        trajs: Dict[int, np.ndarray] = {}
        for tid, pts in bucket.items():
            arr = np.asarray(pts, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                trajs[tid] = arr
        if not trajs:
            raise RuntimeError("No valid trajectories found in frame-based JSON.")
        return trajs

    raise RuntimeError("Unrecognized JSON trajectory format.")

def load_frame_based(json_path: str):
    """
    Strict loader for frame-based JSON.

    Returns:
      frame_list: [timestamp1, timestamp2, ...]
      frame_objects: [ {tid: [x,y], ...}, ... ]  # only objects present in that frame
      tid_frame_history: {tid: [(frame_idx, [x,y]), ...] }  # sparse history with indices
      raw_frames: original list loaded (to allow exporting aligned structure if desired)
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("This file is not in frame-based JSON format (expected a list).")

    frame_list: List[int] = []
    frame_objects: List[Dict[int, List[float]]] = []
    tid_frame_history: Dict[int, List[Tuple[int, List[float]]]] = {}

    for i, frame in enumerate(data):
        ts = frame.get("time_stamp")
        objs = frame.get("object_list", [])
        frame_list.append(ts)

        fr_dict: Dict[int, List[float]] = {}
        for o in objs:
            tid = o.get("track_id")
            pos = o.get("position_3d", {})
            if tid is None:
                continue
            x, y = pos.get("x"), pos.get("y")
            if x is None or y is None:
                continue
            fr_dict[int(tid)] = [float(x), float(y)]
            tid_frame_history.setdefault(int(tid), []).append((i, [float(x), float(y)]))
        frame_objects.append(fr_dict)

    return frame_list, frame_objects, tid_frame_history, data

# --------------------- Geometry & resampling ---------------------
def resample_polyline(xy: np.ndarray, m: int = 32) -> np.ndarray:
    """Resample a polyline to m points by arc length."""
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
    """Return ((cx,cy), width, height, angle_deg) for Gaussian ellipse."""
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
    E = np.c_[width/2*np.cos(t), height/2*np.sin(t)] @ R.T + np.array([cx, cy])
    return E

# ===== ML Model (features, predict, save/load) =====
class MLModel:
    """Route classifier (start_entry -> end_entry) with learned features and progress-aware dynamics."""
    def __init__(self):
        self.entry_centers: Optional[np.ndarray] = None  # (K,2)
        self.entry_ellipses: Optional[List[Tuple]] = None  # [(center,w,h,ang), ...]
        self.pair_to_cid: Dict[Tuple[int,int], int] = {}
        self.cid_to_pair: Dict[int, Tuple[int,int]] = {}
        self.code_centers: Optional[np.ndarray] = None  # (K_code,2)
        self.pca: Optional[PCA] = None  # shape-PCA (2*32 -> N_PCA)
        self.mu: Optional[np.ndarray] = None  # feature mean (D,)
        self.sigma: Optional[np.ndarray] = None  # feature std (D,)
        self.clf: Optional[LogisticRegression] = None  # multinomial softmax
        self.class_step_stats: Dict[int, Tuple[np.ndarray,np.ndarray]] = {}  # cid -> (mu,cov)

        # progress-aware
        self.progress_bins: Optional[np.ndarray] = None  # (B,)
        self.class_progress_stats: Dict[int, Dict[str,np.ndarray]] = {} # {cid: {"mus","covs","counts","avg_len"}}

    # -------- features --------
    def _code_hist(self, dif: np.ndarray) -> np.ndarray:
        if dif.size == 0:
            return np.zeros(self.code_centers.shape[0], dtype=float)
        # nearest codeword per step
        d2 = ((dif[:,None,:] - self.code_centers[None,:,:])**2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        hist = np.bincount(idx, minlength=self.code_centers.shape[0]).astype(float)
        return hist / (hist.sum() + 1e-12)

    def _shape_pca(self, seg: np.ndarray) -> np.ndarray:
        R = resample_polyline(seg, m=32)
        R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-6)
        flat = R.reshape(-1)
        return self.pca.transform(flat[None, :])[0]

    def seg_features(self, seg: np.ndarray) -> np.ndarray:
        dif = np.diff(seg, axis=0)
        spd = np.linalg.norm(dif, axis=1)
        path_len = float(spd.sum()) if len(spd) else 0.0
        net = float(np.linalg.norm(seg[-1]-seg[0]))
        straight = net/(path_len + 1e-12)
        mean_v = float(spd.mean()) if len(spd) else 0.0
        std_v = float(spd.std()) if len(spd) else 0.0
        if len(dif) >= 2:
            th = np.unwrap(np.arctan2(dif[:,1], dif[:,0])); rev_frac = float(np.mean(np.abs(np.diff(th)) > np.deg2rad(120)))
        else:
            rev_frac = 0.0
        centroid = seg.mean(axis=0)
        rg = float(np.sqrt(np.mean(((seg - centroid)**2).sum(axis=1))))
        bbox = float(np.linalg.norm(seg.max(axis=0) - seg.min(axis=0)))
        head = seg[-1]
        dists = np.linalg.norm(self.entry_centers - head[None,:], axis=1) # (K,)
        feat = np.r_[
            np.array([path_len, net, straight, mean_v, std_v, rev_frac, rg, bbox]),
            dists,
            self._code_hist(dif),
            self._shape_pca(seg)
        ]
        return feat

    # -------- predict --------
    def predict_segment_ml(self, seg: np.ndarray, topk: int = 3):
        x = self.seg_features(seg)
        xz = (x - self.mu) / self.sigma
        p = self.clf.predict_proba(xz[None,:])[0]  # (C,)
        order = np.argsort(-p)
        classes = [self.cid_to_pair[int(i)] for i in order[:topk]]
        probs = [float(p[i]) for i in order[:topk]]
        return classes, probs, float(1.0 - p[order[0]])

    # -------- save/load --------
    def save(self, path: str):
        ensure_dir(os.path.dirname(path))
        C = len(self.cid_to_pair)
        cid_to_pair = np.array([self.cid_to_pair[i] for i in range(C)], dtype=int)

        # class step stats
        mus = np.zeros((C,2)); covs = np.zeros((C,2,2))
        for i in range(C):
            m, S = self.class_step_stats.get(i, (np.zeros(2), np.eye(2)*1e-3))
            mus[i] = m; covs[i] = S

        # entry ellipses
        ell = np.array([[e[0][0], e[0][1], e[1], e[2], e[3]] for e in self.entry_ellipses], dtype=float)

        # progress-aware
        if self.progress_bins is None:
            self.progress_bins = np.linspace(0.05, 0.95, 10)
        B = len(self.progress_bins)
        pmus = np.zeros((C,B,2)); pcovs = np.zeros((C,B,2,2)); pcounts = np.zeros((C,B)); avg_len = np.zeros(C)
        for i in range(C):
            st = self.class_progress_stats.get(i, None)
            if st:
                pmus[i] = st["mus"]; pcovs[i] = st["covs"]; pcounts[i] = st["counts"]; avg_len[i] = st["avg_len"]
            else:
                pcovs[i] = np.eye(2)[None,None,:,:]; avg_len[i] = 1.0

        np.savez_compressed(path,
            entry_centers=self.entry_centers,
            entry_ellipses=ell,
            code_centers=self.code_centers,
            pca_mean=self.pca.mean_,
            pca_components=self.pca.components_,
            pca_explained_variance=self.pca.explained_variance_,
            pca_explained_variance_ratio=self.pca.explained_variance_ratio_,
            pca_singular_values=self.pca.singular_values_,
            pca_whiten=self.pca.whiten,
            pca_n=self.pca.n_components_,
            mu=self.mu, sigma=self.sigma,
            clf_coef=self.clf.coef_, clf_inter=self.clf.intercept_, clf_classes=self.clf.classes_,
            cid_to_pair=cid_to_pair,
            step_mus=mus, step_covs=covs,
            progress_bins=self.progress_bins,
            prog_mus=pmus, prog_covs=pcovs, prog_counts=pcounts, class_avg_len=avg_len
        )

    @staticmethod
    def load(path: str) -> "MLModel":
        z = np.load(path, allow_pickle=False)
        m = MLModel()
        m.entry_centers = z["entry_centers"]
        ell = z["entry_ellipses"]; m.entry_ellipses = [((row[0],row[1]), row[2], row[3], row[4]) for row in ell]
        m.code_centers = z["code_centers"]

        # PCA
        pca = PCA(n_components=int(z["pca_n"]), whiten=bool(z["pca_whiten"]))
        pca.mean_ = z["pca_mean"]
        pca.components_ = z["pca_components"]
        # Required for transform()
        pca.explained_variance_ = z["pca_explained_variance"]
        pca.explained_variance_ratio_ = z["pca_explained_variance_ratio"]
        pca.singular_values_ = z["pca_singular_values"]
        # Metadata
        m.pca = pca
        m.mu, m.sigma = z["mu"], z["sigma"]

        # clf
        clf = LogisticRegression(solver="lbfgs", max_iter=1) # dummy
        clf.classes_ = z["clf_classes"]; clf.coef_ = z["clf_coef"]; clf.intercept_ = z["clf_inter"]
        m.clf = clf

        # class id maps
        cid_to_pair = z["cid_to_pair"]; m.cid_to_pair = {int(i): (int(cid_to_pair[i,0]), int(cid_to_pair[i,1])) for i in range(cid_to_pair.shape[0])}
        m.pair_to_cid = {v:k for k,v in m.cid_to_pair.items()}

        # step stats
        mus, covs = z["step_mus"], z["step_covs"]
        for i in range(mus.shape[0]): m.class_step_stats[i] = (mus[i], covs[i])

        # progress-aware
        m.progress_bins = z["progress_bins"]
        pmus, pcovs, pcounts, avg_len = z["prog_mus"], z["prog_covs"], z["prog_counts"], z["class_avg_len"]
        for i in range(pmus.shape[0]):
            m.class_progress_stats[i] = {"mus": pmus[i], "covs": pcovs[i], "counts": pcounts[i], "avg_len": float(avg_len[i])}
        return m

# ===== Training pipeline & report =====
def auto_tune_and_train_ml(trajs: Dict[int,np.ndarray],
    K_ENTRIES: int = 5, K_CODE: int = 16, N_PCA: int = 6,
    prefix_fracs: List[float] = [0.15,0.25,0.35,0.5,0.7,0.9],
    seed: int = 0) -> MLModel:

    rng = np.random.default_rng(seed)
    model = MLModel()

    # 1) Entry areas
    starts = np.vstack([arr[0] for arr in trajs.values()])
    ends = np.vstack([arr[-1] for arr in trajs.values()])
    P = np.vstack([starts, ends])
    kmeans_entry = KMeans(n_clusters=K_ENTRIES, random_state=seed, n_init="auto").fit(P)
    centers = kmeans_entry.cluster_centers_
    model.entry_centers = centers

    # ellipses
    labels = kmeans_entry.labels_
    ell = []
    for ci in range(K_ENTRIES):
        pts = P[labels==ci]
        if len(pts) >= 2: center, w, h, ang = ellipse_params(pts, n_std=2.5)
        else: center = tuple(centers[ci]); w=h=0.2; ang=0.0
        ell.append((center, w, h, ang))
    model.entry_ellipses = ell

    # 2) True route class per trajectory
    traj_pair = {}
    for tid, arr in trajs.items():
        sc = int(np.argmin(((arr[0]-centers)**2).sum(axis=1)))
        ec = int(np.argmin(((arr[-1]-centers)**2).sum(axis=1)))
        traj_pair[tid] = (sc, ec)
    pair_to_cid = {p:i for i,p in enumerate(sorted(set(traj_pair.values())))}
    model.pair_to_cid = pair_to_cid
    model.cid_to_pair = {i:p for p,i in pair_to_cid.items()}

    # 3) Codebook over step vectors
    steps = [np.diff(arr, axis=0) for arr in trajs.values() if len(arr)>=2]
    steps = np.vstack([s for s in steps if len(s)]) if steps else np.zeros((1,2))
    sp = np.linalg.norm(steps, axis=1)
    lo, hi = np.quantile(sp, [0.01, 0.99]) if len(sp)>10 else (sp.min(), sp.max()+1e-6)
    base = steps[(sp>=lo)&(sp<=hi)] if len(sp)>10 else steps
    model.code_centers = KMeans(n_clusters=K_CODE, random_state=seed, n_init="auto").fit(base).cluster_centers_

    # 4) PCA on shapes
    resampled = []
    for tid, arr in list(trajs.items())[:min(120, len(trajs))]:
        R = resample_polyline(arr, m=32)
        R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-6)
        resampled.append(R.reshape(-1))
    Xp = np.vstack(resampled) if resampled else np.zeros((1,64))
    pca = PCA(n_components=N_PCA, whiten=False, random_state=seed).fit(Xp)
    model.pca = pca

    # 5) Build training set of prefixes
    X_list, y_list = [], []
    for tid, arr in trajs.items():
        cid = pair_to_cid[traj_pair[tid]]
        for f in prefix_fracs:
            n = max(5, int(len(arr)*f))
            if n >= len(arr): n = len(arr)-1
            if n <= 4: continue
            X_list.append(model.seg_features(arr[:n]))
            y_list.append(cid)
    X = np.vstack(X_list); y = np.array(y_list, dtype=int)
    model.mu = X.mean(axis=0); model.sigma = X.std(axis=0)+1e-6
    Xz = (X - model.mu)/model.sigma

    # 6) Multinomial logistic regression
    clf = LogisticRegression(solver="lbfgs", max_iter=400, random_state=seed)
    clf.fit(Xz, y)
    model.clf = clf

    # 7) Per-class step dynamics
    class_stats = {}
    for tid, arr in trajs.items():
        cid = pair_to_cid[traj_pair[tid]]
        dif = np.diff(arr, axis=0)
        if len(dif)==0: continue
        class_stats.setdefault(cid, []).append(dif)
    for cid, lst in class_stats.items():
        V = np.vstack(lst)
        m = V.mean(axis=0); cov = np.cov(V.T) if V.shape[0]>1 else np.eye(2)*1e-3
        model.class_step_stats[cid] = (m, cov)

    # 8) Progress-aware dynamics (10 bins)
    B = 10; bins = np.linspace(0.05, 0.95, B)
    model.progress_bins = bins
    prog = {cid: {"sum": np.zeros((B,2)), "cross": np.zeros((B,2,2)), "count": np.zeros(B), "lens": []}
            for cid in range(len(model.cid_to_pair))}
    for tid, arr in trajs.items():
        cid = pair_to_cid[traj_pair[tid]]
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        if len(d)==0: continue
        s = np.r_[0, np.cumsum(d)]
        if s[-1] < 1e-8: continue
        dif = np.diff(arr, axis=0)
        for i in range(len(dif)):
            p = (s[i] / s[-1]); b = int(np.clip(np.round((B-1)*p),0,B-1))
            prog[cid]["sum"][b] += dif[i]
            prog[cid]["cross"][b] += np.outer(dif[i], dif[i])
            prog[cid]["count"][b] += 1
        prog[cid]["lens"].append(s[-1])
    for cid in range(len(model.cid_to_pair)):
        cnt = prog[cid]["count"]; mus = np.zeros((B,2)); covs = np.zeros((B,2,2))
        for b in range(B):
            if cnt[b] > 1:
                mu_b = prog[cid]["sum"][b]/cnt[b]
                S_b = prog[cid]["cross"][b]/cnt[b] - np.outer(mu_b, mu_b)
                mus[b]=mu_b; covs[b]=S_b+1e-6*np.eye(2)
            else:
                covs[b]=np.eye(2)*1e-3
        avg_len = np.mean(prog[cid]["lens"]) if prog[cid]["lens"] else 1.0
        model.class_progress_stats[cid]={"mus":mus,"covs":covs,"counts":cnt,"avg_len":float(avg_len)}
    return model

def evaluate_prefix_accuracy(trajs: Dict[int,np.ndarray], model: MLModel,
    prefix_fracs: List[float] = [0.15,0.25,0.35,0.5,0.7,0.9],
    topk: int = 3):
    # True class per trajectory based on entries
    true_cid = {}
    for tid, arr in trajs.items():
        sc = int(np.argmin(((arr[0]-model.entry_centers)**2).sum(axis=1)))
        ec = int(np.argmin(((arr[-1]-model.entry_centers)**2).sum(axis=1)))
        true_cid[tid] = model.pair_to_cid[(sc,ec)]

    results = []
    for f in prefix_fracs:
        c1=cK=tot=0
        for tid, arr in trajs.items():
            n=max(6,int(len(arr)*f)); n=min(n,len(arr)-1)
            if n<=5: continue
            classes, probs, unc = model.predict_segment_ml(arr[:n], topk=topk)
            pred_cids = [model.pair_to_cid[c] for c in classes]
            t = true_cid[tid]
            tot+=1
            if pred_cids[0]==t: c1+=1
            if t in pred_cids: cK+=1
        if tot>0: results.append((f, c1/tot, cK/tot, tot))
    return results

def save_accuracy_plot(results, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    fracs=[r[0]*100 for r in results]; top1=[r[1]*100 for r in results]; topk=[r[2]*100 for r in results]; n=[r[3] for r in results]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(fracs, top1, marker='o', label='Top-1'); ax.plot(fracs, topk, marker='s', label='Top-3')
    for x,y,c in zip(fracs, top1, n): ax.text(x, y+1.5, f"n={c}", ha='center', fontsize=8)
    ax.set_xlabel("Prefix length (%)"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Route accuracy vs prefix")
    ax.set_ylim(0,105); ax.grid(True, ls='--', alpha=0.4); ax.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

# ===== Micro maneuvers detection with FULL tuning =====
def compute_window_features(seg: np.ndarray, rev_angle_deg: float):
    dif = np.diff(seg, axis=0)
    spd = np.linalg.norm(dif, axis=1)
    mean_v = float(spd.mean()) if len(spd) else 0.0
    std_v = float(spd.std()) if len(spd) else 0.0
    if len(dif) >= 2:
        th = np.unwrap(np.arctan2(dif[:,1], dif[:,0]))
        rev_frac = float(np.mean(np.abs(np.diff(th)) > np.deg2rad(rev_angle_deg)))
    else:
        rev_frac = 0.0
    centroid = seg.mean(axis=0)
    rg = float(np.sqrt(np.mean(((seg-centroid)**2).sum(axis=1))))
    bbox = float(np.linalg.norm(seg.max(axis=0) - seg.min(axis=0)))
    return {"mean_v": mean_v, "std_v": std_v, "rev_frac": rev_frac, "rg": rg, "bbox": bbox}

def auto_tune_mode_thresholds(trajs: Dict[int,np.ndarray], win: int = 25, rev_angle_deg: float = 120.0) -> dict:
    feats = {"mean_v": [], "std_v": [], "rev_frac": [], "rg": [], "bbox": []}
    for arr in trajs.values():
        if len(arr) < max(6, win): continue
        for i in range(win, len(arr)+1):
            f = compute_window_features(arr[i-win:i], rev_angle_deg)
            for k in feats: feats[k].append(f[k])
    if not all(len(v)>0 for v in feats.values()):
        # safe defaults
        return {"stationary_mean_v_max":0.05,"stationary_rg_max":0.2,"stationary_bbox_max":0.2,
                "baf_bbox_max":1.5,"baf_rev_frac_min":0.2}
    P = {k: np.percentile(v, [5,25,50,75,95]) for k,v in feats.items()}
    T = {
        "stationary_mean_v_max": float(P["mean_v"][1]),
        "stationary_rg_max": float(P["rg"][1]),
        "stationary_bbox_max": float(P["bbox"][1]),
        "baf_bbox_max": float(P["bbox"][2]),
        "baf_rev_frac_min": float(P["rev_frac"][3]),
    }
    return T

def motion_mode_probs(seg: np.ndarray, T: dict,
    rev_angle_deg: float = 120.0,
    baf_sensitivity: float = 1.0,
    stationary_sens: float = 1.0,
    bbox_sens: float = 1.0) -> dict:

    f = compute_window_features(seg, rev_angle_deg)
    # Stationary score (scaled by stationary_sens & bbox_sens)
    s_stat = 0.0
    s_stat += max(0.0, 1 - f["mean_v"]/(stationary_sens*T["stationary_mean_v_max"] + 1e-6))
    s_stat += max(0.0, 1 - f["rg"] /(stationary_sens*T["stationary_rg_max"] + 1e-6))
    s_stat += max(0.0, 1 - f["bbox"] /(bbox_sens *T["stationary_bbox_max"] + 1e-6))

    # Back-and-forth score (scaled by baf_sensitivity & bbox_sens)
    baf_tight = max(0.0, 1 - f["bbox"]/(bbox_sens*T["baf_bbox_max"] + 1e-6))
    baf_turn = max(0.0, (f["rev_frac"] - T["baf_rev_frac_min"]) / max(1e-6, 1 - T["baf_rev_frac_min"]))
    s_baf = baf_sensitivity * (baf_tight + baf_turn)

    # Moving baseline
    s_mov = 0.5 + 0.5*max(0.0, f["mean_v"]/max(T["stationary_mean_v_max"],1e-3))

    raw = np.array([s_stat, s_baf, s_mov])
    p = np.exp(raw - raw.max()); p = p/p.sum()
    return {"stationary": float(p[0]), "loading": float(p[1]), "moving": float(p[2])}

class OnlineClassifier:
    """Combines route class and micro mode with hysteresis and live tuning."""
    def __init__(self, model: MLModel, T: dict,
        window: int = 25, enter_baf: int = 3, exit_baf: int = 5,
        rev_angle_deg: float = 120.0,
        baf_sensitivity: float = 1.0,
        stationary_sens: float = 1.0,
        bbox_sens: float = 1.0):

        self.model = model; self.T = T
        self.window = window
        self.enter_baf = enter_baf; self.exit_baf = exit_baf
        self.rev_angle_deg = rev_angle_deg
        self.baf_sensitivity = baf_sensitivity
        self.stationary_sens = stationary_sens
        self.bbox_sens = bbox_sens
        self.mode = "moving"; self._on=0; self._off=0

    def update(self, prefix: np.ndarray):
        classes, probs, unc = self.model.predict_segment_ml(prefix)
        route_pair = classes[0]; route_prob = probs[0]
        seg = prefix[-max(6, self.window):]
        pm = motion_mode_probs(seg, self.T, self.rev_angle_deg,
                               self.baf_sensitivity, self.stationary_sens, self.bbox_sens)
        raw_mode = max(pm, key=pm.get)
        if raw_mode == "loading":
            self._on += 1; self._off = 0
            if self._on >= self.enter_baf: self.mode = "loading"
        else:
            if self.mode == "loading":
                self._off += 1
                if self._off >= self.exit_baf:
                    self.mode = raw_mode; self._on=self._off=0
            else:
                self.mode = raw_mode

        display_label = ("loading" if self.mode=="loading" else f"route:{route_pair[0]}→{route_pair[1]}")
        display_conf = (pm["loading"] if self.mode=="loading" else route_prob)
        return {"route_pair": route_pair, "route_prob": route_prob, "route_uncertainty": unc,
                "mode": self.mode, "mode_probs": pm,
                "display_label": display_label, "display_confidence": display_conf}

# ===== Forecasting (progress-aware optional) =====
def blend_velocity(mu_local: np.ndarray, cov_local: np.ndarray,
    mu_class: np.ndarray, cov_class: np.ndarray,
    seg_len: int, kappa: float = 0.002):
    alpha = 1.0 - np.exp(-kappa*max(1, seg_len))
    mu = alpha*mu_local + (1-alpha)*mu_class
    cov = alpha*cov_local + (1-alpha)*cov_class
    return mu, cov, alpha

def class_progress_velocity(model: MLModel, cid: int, est_progress: float):
    bins = model.progress_bins
    bidx = int(np.clip(np.round((len(bins)-1) * est_progress), 0, len(bins)-1))
    st = model.class_progress_stats.get(cid, None)
    if not st:
        return model.class_step_stats.get(cid, (np.zeros(2), np.eye(2)*1e-3))
    return st["mus"][bidx], st["covs"][bidx]

def estimate_progress_for_class(prefix: np.ndarray, model: MLModel, cid: int):
    d = np.linalg.norm(np.diff(prefix, axis=0), axis=1); obs_len = float(d.sum()) if len(d) else 0.0
    avg_len = model.class_progress_stats.get(cid, {}).get("avg_len", 1.0)
    return float(np.clip(obs_len / max(1e-6, avg_len), 0.0, 1.0))

def forecast_particles(prefix: np.ndarray,
    online,
    horizon: int,
    n_particles: int = 1500,
    seed: int = 0,
    use_progress_dyn: bool = False,
    # --- Loading behavior controls ---
    loading_drift_scale: float = 0.4,
    loading_cov_scale: float = 1.2,
    loading_anchor_alpha: float = 0.7, # 0..1: 1=use centroid, 0=use last point
    # --- Stationary behavior controls ---
    stationary_cov_scale: float = 0.25,
    stationary_eps: float = 5e-5,
    # --- Numeric safety ---
    cov_floor: float = 1e-6):

    """
    Forecast particle endpoints `parts` at `horizon` steps ahead, conditioned on:
      - micro-mode from OnlineClassifier (moving / loading / other),
      - local kinematics from recent window,
      - class priors (and optionally progress-aware stats).
    """
    rng = np.random.default_rng(seed)

    # ---------- Local kinematics from recent window ----------
    w = max(6, online.window)
    seg = prefix[-w:]
    dif = np.diff(seg, axis=0)
    if dif.shape[0] == 0:
        mu_local = np.zeros(2)
        cov_local = np.eye(2) * 1e-3
    elif dif.shape[0] == 1:
        mu_local = dif[0]
        cov_local = np.eye(2) * 1e-3
    else:
        mu_local = dif.mean(axis=0)
        cov_local = np.cov(dif.T)
    # Safety: small floor to keep PSD
    cov_local = cov_local + 1e-12 * np.eye(2)

    # ---------- Update classifier and pull class stats ----------
    info = online.update(prefix)
    cid = online.model.pair_to_cid[info["route_pair"]]

    # Base class stats (average over class)
    mu_c, cov_c = online.model.class_step_stats.get(
        cid, (np.zeros(2), np.eye(2) * 1e-3)
    )

    # Optionally progress-conditioned stats
    if use_progress_dyn:
        try:
            prog = estimate_progress_for_class(prefix, online.model, cid)
            mu_c, cov_c = class_progress_velocity(online.model, cid, prog)
        except Exception:
            # Safe fallback if progress estimation not available/robust
            mu_c, cov_c = online.model.class_step_stats.get(
                cid, (np.zeros(2), np.eye(2) * 1e-3)
            )

    # ---------- Select dynamics by micro-mode ----------
    mode = info["mode"]
    if mode == "moving":
        # Blend local behavior with class priors
        mu, cov, alpha = blend_velocity(mu_local, cov_local, mu_c, cov_c, len(prefix))
        x0 = prefix[-1]
    elif mode == "loading":
        # online.model.class_loading_stats[cid] = (mu_load, cov_load)
        mu_load, cov_load = online.model.__dict__.get("class_loading_stats", {}).get(
            cid, (mu_c, cov_c)
        )
        # Blend local with (loading-aware) class stats to keep directional structure
        mu_blend, cov_blend, alpha = blend_velocity(mu_local, cov_local, mu_load, cov_load, len(prefix))
        # Loading is "slow drift + jitter"
        mu = loading_drift_scale * mu_blend
        cov = loading_cov_scale * cov_blend + 1e-4 * np.eye(2)
        # Anchor start near the centroid to avoid last-sample bias,
        # but keep some weight on the last point to preserve directionality.
        centroid = seg.mean(axis=0)
        x_last = prefix[-1]
        x0 = loading_anchor_alpha * centroid + (1.0 - loading_anchor_alpha) * x_last
    else:
        # Other non-moving modes (true stationary, parked, etc.)
        centroid = seg.mean(axis=0)
        x0 = centroid
        mu = np.zeros(2)
        cov = stationary_cov_scale * cov_local + stationary_eps * np.eye(2)

    # Safety floor for sampling
    cov = cov + cov_floor * np.eye(2)

    # ---------- Monte Carlo rollout (endpoints only) ----------
    parts = np.tile(x0, (n_particles, 1))
    if horizon > 0:
        steps = rng.multivariate_normal(mean=mu, cov=cov, size=(horizon, n_particles))
        # Sum increments across time for endpoints
        total_increments = steps.sum(axis=0)
        parts = parts + total_increments

    return info, parts

# ---- Overlay heatmap + CI95 on an existing Axes (main view) ----
def draw_heatmap_and_ci95_on_ax(ax, parts: np.ndarray, label="CI95", cmap="magma"):
    """
    Draw a smooth heatmap and a CI95 ellipse without rectangular artifacts.
    """
    # ---- Heatmap ----
    H2, xe, ye = np.histogram2d(parts[:,0], parts[:,1], bins=80)
    ax.pcolormesh(
        xe, ye, H2.T,
        cmap=cmap,
        shading="auto",
        alpha=0.70,
        edgecolors='none'
    )
    # ---- Compute CI95 ellipse ----
    mu = parts.mean(axis=0)
    cov = np.cov(parts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:,order]
    # 95% quantile (Chi^2(2) = 5.991)
    k = 5.991
    r1 = np.sqrt(vals[0] * k)
    r2 = np.sqrt(vals[1] * k)
    angle = np.arctan2(vecs[1,0], vecs[0,0])
    t = np.linspace(0, 2*np.pi, 300)
    ellipse = np.column_stack((r1*np.cos(t), r2*np.sin(t))) @ vecs.T + mu
    ax.plot(
        ellipse[:,0],
        ellipse[:,1],
        color='cyan',
        lw=2,
        label=label
    )
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_aspect('equal', 'box')

### Heatmap
def plot_future_heatmap_with_CI95(prefix, online, horizon=1,
    n_particles=2000, seed=0,
    use_progress_dyn=False,
    ax=None, title_prefix="H="):

    info, parts = forecast_particles(
        prefix, online, horizon=horizon,
        n_particles=n_particles,
        seed=seed,
        use_progress_dyn=use_progress_dyn
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.figure

    H2, xe, ye = np.histogram2d(parts[:,0], parts[:,1], bins=120)
    ax.pcolormesh(xe, ye, H2.T, cmap="magma", shading="auto")

    mu = parts.mean(axis=0)
    cov = np.cov(parts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    k = 5.991
    r1 = np.sqrt(k * vals[0]); r2 = np.sqrt(k * vals[1])
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    t = np.linspace(0, 2*np.pi, 300)
    ellipse_points = np.c_[r1*np.cos(t), r2*np.sin(t)] @ vecs.T + mu
    ax.plot(ellipse_points[:,0], ellipse_points[:,1],
            color='cyan', lw=2, label="CI95 ellipse")
    x0 = prefix[-1]
    ax.scatter([x0[0]],[x0[1]], c='yellow', edgecolor='k',
               s=70, zorder=5, label='Current')
    lbl = info['display_label']
    conf = info['display_confidence']
    ax.set_title(f"{title_prefix}{horizon} — {lbl} (conf={conf:.2f})")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc='best')
    fig.tight_layout()
    return fig, ax

# ===== Online state for multi-objects (frame-based) =====
class OnlinePool:
    """
    Keeps a separate OnlineClassifier per track_id and replays history when needed
    to keep hysteresis consistent as the slider moves.
    """
    def __init__(self, model: MLModel, T: dict, window: int = 25,
                 enter_baf: int = 3, exit_baf: int = 5,
                 rev_angle_deg: float = 120.0,
                 baf_sensitivity: float = 1.0,
                 stationary_sens: float = 1.0,
                 bbox_sens: float = 1.0):
        self.model = model
        self.T = T
        self.cfg = dict(window=window, enter_baf=enter_baf, exit_baf=exit_baf,
                        rev_angle_deg=rev_angle_deg, baf_sensitivity=baf_sensitivity,
                        stationary_sens=stationary_sens, bbox_sens=bbox_sens)
        # tid -> (OnlineClassifier, last_frame_idx_processed)
        self.store: Dict[int, Tuple[OnlineClassifier, int]] = {}

    def _new_online(self) -> OnlineClassifier:
        return OnlineClassifier(self.model, self.T, **self.cfg)

    def update_to_frame(self, tid: int, tid_hist: List[Tuple[int, List[float]]], target_frame: int):
        """
        Ensure the online classifier for tid has processed up to target_frame.
        tid_hist is sorted by frame_idx. Replays only what’s necessary.
        """
        if not tid_hist:
            return None

        # Create if not exists
        if tid not in self.store:
            self.store[tid] = (self._new_online(), -1)

        online, last_idx = self.store[tid]

        # If we jumped backward, rebuild state
        if last_idx > target_frame:
            online = self._new_online()
            last_idx = -1

        # Replay from next unprocessed point up to target_frame (inclusive)
        # Build progressive prefixes of the object's appearances (not every frame).
        xs = []
        for (fi, xy) in tid_hist:
            if fi > target_frame:
                break
            xs.append(xy)

        if len(xs) >= 2:
            prefix = np.asarray(xs, dtype=float)
            # Call once with the full prefix (update relies on internal window/hysteresis)
            online.update(prefix)
            self.store[tid] = (online, target_frame)
            return online, prefix
        else:
            # Not enough history to predict
            self.store[tid] = (online, target_frame)
            return online, None

    def reset(self):
        self.store.clear()

# ===== GUI (PyQt6) with frame-mode support and export =====
def run_gui(trajs: Dict[int,np.ndarray], model: Optional[MLModel], online: Optional[OnlineClassifier],
            K_ENTRIES: int, K_CODE: int, N_PCA: int, WIN: int, use_progress_dyn: bool,
            frame_list: Optional[List[int]] = None,
            frame_objects: Optional[List[Dict[int, List[float]]]] = None,
            tid_frame_history: Optional[Dict[int, List[Tuple[int, List[float]]]]] = None,
            orig_frames: Optional[list] = None,
            source_json_path: Optional[str] = None):

    try:
        from PyQt6 import QtCore, QtWidgets
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
    except Exception:
        print("PyQt6 is required: pip install PyQt6 matplotlib"); raise

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    class StartDialog(QtWidgets.QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Load or Train Model")
            v = QtWidgets.QVBoxLayout(self)
            self.list = QtWidgets.QListWidget()
            v.addWidget(QtWidgets.QLabel("Saved models (newest first):")); v.addWidget(self.list)
            h = QtWidgets.QHBoxLayout(); self.btn_load=QtWidgets.QPushButton("Load selected")
            self.btn_train=QtWidgets.QPushButton("Train new"); self.btn_cancel=QtWidgets.QPushButton("Cancel")
            h.addWidget(self.btn_load); h.addWidget(self.btn_train); h.addStretch(1); h.addWidget(self.btn_cancel); v.addLayout(h)
            self.btn_cancel.clicked.connect(self.reject)
            self.btn_load.clicked.connect(self.accept_load)
            self.btn_train.clicked.connect(self.accept_train)
            self.files = list_saved_models()
            for p in self.files:
                ts = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
                self.list.addItem(f"{os.path.basename(p)} ({ts})")
            self.choice=None
        def accept_load(self):
            row=self.list.currentRow()
            if row<0: QtWidgets.QMessageBox.warning(self,"No selection","Select a model."); return
            self.choice=('load', self.files[row]); self.accept()
        def accept_train(self): self.choice=('train', None); self.accept()

    if model is None or online is None:
        dlg = StartDialog()
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted: return
        ch, path = dlg.choice
        if ch=='load':
            model = MLModel.load(path)
            T = auto_tune_mode_thresholds(trajs, win=WIN, rev_angle_deg=120.0)
            online = OnlineClassifier(model, T, window=WIN, enter_baf=3, exit_baf=5,
                                      rev_angle_deg=120.0, baf_sensitivity=1.0, stationary_sens=1.0, bbox_sens=1.0)
        else:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                model = auto_tune_and_train_ml(trajs, K_ENTRIES, K_CODE, N_PCA)
                savep = timestamped_model_path(); model.save(savep)
                QtWidgets.QMessageBox.information(None,"Model saved",f"Saved:\n{savep}")
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            T = auto_tune_mode_thresholds(trajs, win=WIN, rev_angle_deg=120.0)
            online = OnlineClassifier(model, T, window=WIN, enter_baf=3, exit_baf=5,
                                      rev_angle_deg=120.0, baf_sensitivity=1.0, stationary_sens=1.0, bbox_sens=1.0)

    class TrajPlayer(QtWidgets.QMainWindow):
        """
        Trajectory monitor with dual modes:

        1) Trajectory-mode (original): single trajectory scrub by prefix.
        2) Frame-mode: time-synced frames, multiple objects per frame with selection and export.

        - Maintains 1:1 aspect plotting
        - Optionally overlay forecast for selected object
        - Table of all active objects' predictions per frame
        - Export predictions to new JSON file
        """
        def __init__(self, trajs, model, online,
                     bg_image_path: str = None,
                     bg_extent: tuple = None, # (xmin, xmax, ymin, ymax)
                     bg_origin: str = 'lower',
                     bg_alpha: float = 0.85,
                     frame_list: Optional[List[int]] = None,
                     frame_objects: Optional[List[Dict[int, List[float]]]] = None,
                     tid_frame_history: Optional[Dict[int, List[Tuple[int, List[float]]]]] = None,
                     orig_frames: Optional[list] = None,
                     source_json_path: Optional[str] = None):

            super().__init__()
            self.setWindowTitle("Trajectory Monitor — Warehouse forklift")
            self.trajs = trajs
            self.model = model
            self.online_single = online  # used in trajectory mode
            self.use_progress_dyn = use_progress_dyn
            self.tids = sorted(trajs.keys()) if trajs else []
            self.cur_tid = self.tids[0] if self.tids else None
            self.cur_idx = 6

            # --- Frame mode data ---
            self.frame_list = frame_list
            self.frame_objects = frame_objects
            self.tid_frame_history = tid_frame_history
            self.orig_frames = orig_frames
            self.source_json_path = source_json_path
            self.use_frame_mode = frame_list is not None and frame_objects is not None
            self.current_frame = 0

            # Keep per-object online states in frame-mode
            self.T = auto_tune_mode_thresholds(self.trajs, win=self.online_single.window, rev_angle_deg=self.online_single.rev_angle_deg)
            self.pool = OnlinePool(self.model, self.T,
                                   window=self.online_single.window,
                                   enter_baf=self.online_single.enter_baf,
                                   exit_baf=self.online_single.exit_baf,
                                   rev_angle_deg=self.online_single.rev_angle_deg,
                                   baf_sensitivity=self.online_single.baf_sensitivity,
                                   stationary_sens=self.online_single.stationary_sens,
                                   bbox_sens=self.online_single.bbox_sens)

            # --- Background config ---
            self.bg_image_path = bg_image_path
            self.bg_extent = bg_extent
            self.bg_origin = bg_origin
            self.bg_alpha = bg_alpha
            self._bg_img = None
            self._view_extent = None

            # --- Timer ---
            self._was_playing = False
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(80)
            self.timer.timeout.connect(self.step_forward)

            # --- Layout ---
            main = QtWidgets.QWidget(self)
            self.setCentralWidget(main)
            root = QtWidgets.QHBoxLayout(main)

            # Left: plot + controls
            left = QtWidgets.QVBoxLayout()
            self.fig_main = Figure(figsize=(4, 6), tight_layout=True)
            self.ax_main = self.fig_main.add_subplot(111)
            self.ax_main.set_aspect('equal', adjustable='datalim')
            try: self.ax_main.set_box_aspect(1)
            except Exception: pass
            self.canvas_main = FigureCanvas(self.fig_main)
            left.addWidget(self.canvas_main, stretch=6)

            # Controls row
            ctrl = QtWidgets.QHBoxLayout()
            self.cb = QtWidgets.QComboBox()
            [self.cb.addItem(str(t)) for t in self.tids]
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
            ctrl.addWidget(self.btn_play)
            ctrl.addWidget(self.btn_b)
            ctrl.addWidget(self.btn_f)
            ctrl.addWidget(self.btn_r)
            ctrl.addStretch(1)
            left.addLayout(ctrl)

            # Frame-mode: object picker + save predictions
            ctrl3 = QtWidgets.QHBoxLayout()
            self.cb_objects = QtWidgets.QComboBox()
            self.cb_objects.currentIndexChanged.connect(self.update_all)
            self.btn_save_pred = QtWidgets.QPushButton("Save predictions JSON")
            self.btn_save_pred.clicked.connect(self.save_predictions_dialog)
            if self.use_frame_mode:
                ctrl3.addWidget(QtWidgets.QLabel("Object:"))
                ctrl3.addWidget(self.cb_objects)
                ctrl3.addStretch(1)
                ctrl3.addWidget(self.btn_save_pred)
                left.addLayout(ctrl3)

            # Prefix slider row (used for both modes, but meaning differs)
            ctrl2 = QtWidgets.QHBoxLayout()
            ctrl2.addWidget(QtWidgets.QLabel("Position"))
            self.sld_prefix = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.lbl_prefix = QtWidgets.QLabel("")  # set later
            self.sld_prefix.valueChanged.connect(self._on_prefix_slider_changed)
            self.sld_prefix.sliderPressed.connect(self._on_prefix_slider_pressed)
            self.sld_prefix.sliderReleased.connect(self._on_prefix_slider_released)
            ctrl2.addWidget(self.sld_prefix, stretch=1)
            ctrl2.addWidget(self.lbl_prefix)
            left.addLayout(ctrl2)

            # Right: route + micro + table
            right = QtWidgets.QVBoxLayout()
            self.fig_route = Figure(figsize=(4, 3), tight_layout=True)
            self.ax_route = self.fig_route.add_subplot(111)
            self.canvas_route = FigureCanvas(self.fig_route)
            right.addWidget(self.canvas_route, stretch=1)

            self.fig_micro = Figure(figsize=(4, 3), tight_layout=True)
            self.ax_micro = self.fig_micro.add_subplot(111)
            self.canvas_micro = FigureCanvas(self.fig_micro)
            right.addWidget(self.canvas_micro, stretch=1)

            # Table of active objects (frame-mode)
            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels(["track_id","label","conf","mode","p(mov)","p(load)","p(stat)"])
            self.table.horizontalHeader().setStretchLastSection(True)
            if self.use_frame_mode:
                right.addWidget(self.table, stretch=1)

            # Tuning panel
            pane = QtWidgets.QGroupBox("Maneuvers Tuning")
            form = QtWidgets.QFormLayout(pane)
            self.sld_baf = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.sld_baf.setRange(5, 30)
            self.sld_baf.setValue(int(self.online_single.baf_sensitivity*10))
            self.lbl_baf = QtWidgets.QLabel(f"{self.online_single.baf_sensitivity:.2f}")
            self.sld_baf.valueChanged.connect(lambda _: self._update_sens())
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.sld_baf)
            row.addWidget(self.lbl_baf)
            form.addRow("Loading sensitivity (0.5-3.0)", row)

            self.spn_rev = QtWidgets.QSpinBox()
            self.spn_rev.setRange(60, 180)
            self.spn_rev.setValue(int(self.online_single.rev_angle_deg))
            self.spn_rev.valueChanged.connect(self._apply_tuning)
            form.addRow("Reversal angle (deg)", self.spn_rev)

            self.spn_win = QtWidgets.QSpinBox()
            self.spn_win.setRange(5, 60)
            self.spn_win.setValue(self.online_single.window)
            self.spn_win.valueChanged.connect(self._apply_tuning)
            form.addRow("Window size", self.spn_win)

            self.spn_enter = QtWidgets.QSpinBox()
            self.spn_enter.setRange(1, 20)
            self.spn_enter.setValue(self.online_single.enter_baf)
            self.spn_enter.valueChanged.connect(self._apply_tuning)
            form.addRow("Loading enter frames", self.spn_enter)

            self.spn_exit = QtWidgets.QSpinBox()
            self.spn_exit.setRange(1, 20)
            self.spn_exit.setValue(self.online_single.exit_baf)
            self.spn_exit.valueChanged.connect(self._apply_tuning)
            form.addRow("Loading exit frames", self.spn_exit)

            self.sld_stat = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.sld_stat.setRange(5, 30)
            self.sld_stat.setValue(int(self.online_single.stationary_sens*10))
            self.lbl_stat = QtWidgets.QLabel(f"{self.online_single.stationary_sens:.2f}")
            self.sld_stat.valueChanged.connect(self._update_sens)
            row2 = QtWidgets.QHBoxLayout()
            row2.addWidget(self.sld_stat)
            row2.addWidget(self.lbl_stat)
            form.addRow("Stationary sensitivity (0.5-3.0)", row2)

            self.sld_bbox = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.sld_bbox.setRange(5, 30)
            self.sld_bbox.setValue(int(self.online_single.bbox_sens*10))
            self.lbl_bbox = QtWidgets.QLabel(f"{self.online_single.bbox_sens:.2f}")
            self.sld_bbox.valueChanged.connect(self._update_sens)
            row3 = QtWidgets.QHBoxLayout()
            row3.addWidget(self.sld_bbox)
            row3.addWidget(self.lbl_bbox)
            form.addRow("BBox tightness sensitivity (0.5-3.0)", row3)

            # Toggle: show forecast overlay
            self.chk_overlay = QtWidgets.QCheckBox("Show forecasted future positions (5 frames ahead)")
            self.chk_overlay.setChecked(True)
            self.chk_overlay.stateChanged.connect(self.update_all)
            form.addRow(self.chk_overlay)

            # Popup heatmaps (uses selected object in frame-mode)
            self.btn_heat = QtWidgets.QPushButton("Show heatmaps (H=1,2,3)")
            self.btn_heat.clicked.connect(self.show_heatmaps)
            form.addRow(self.btn_heat)

            right.addWidget(pane, stretch=0)

            root.addLayout(left, stretch=3)
            root.addLayout(right, stretch=2)

            # Prepare background, slider, etc.
            self._prepare_background_config()
            self._setup_prefix_slider()
            self.update_all()

        # ---------- Background helpers ----------
        def _data_bounds(self):
            """Return (xmin, xmax, ymin, ymax) across all trajectories with small padding."""
            if self.use_frame_mode and self.frame_objects:
                all_pts = []
                for fr in self.frame_objects:
                    for xy in fr.values():
                        all_pts.append(xy)
                if not all_pts:
                    all_pts = [np.zeros(2)]
                all_pts = np.asarray(all_pts, dtype=float)
            else:
                all_pts = np.vstack(list(self.trajs.values())) if self.trajs else np.zeros((1,2))
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            pad_x = 0.05 * (xmax - xmin + 1e-9)
            pad_y = 0.05 * (ymax - ymin + 1e-9)
            return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y

        def _square_from_bounds(self, xmin, xmax, ymin, ymax):
            """Expand bounds to a square extent centered at the box center."""
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            span = max(xmax - xmin, ymax - ymin)
            return (cx - 0.5 * span, cx + 0.5 * span, cy - 0.5 * span, cy + 0.5 * span)

        def _prepare_background_config(self):
            if self.bg_extent is None:
                xmin, xmax, ymin, ymax = self._data_bounds()
                self._view_extent = self._square_from_bounds(xmin, xmax, ymin, ymax)
            else:
                xmin, xmax, ymin, ymax = self.bg_extent
                self._view_extent = self._square_from_bounds(xmin, xmax, ymin, ymax)
            if self.bg_image_path:
                try:
                    self._bg_img = mpimg.imread(self.bg_image_path)
                except Exception as e:
                    print(f"[TrajPlayer] Could not load background image: {e}")
                    self._bg_img = None

        # ---------- Slider helpers ----------
        def _setup_prefix_slider(self):
            if self.use_frame_mode:
                minv = 0
                maxv = max(0, len(self.frame_list)-1)
                self.sld_prefix.blockSignals(True)
                self.sld_prefix.setRange(minv, maxv)
                page = max(1, (maxv - minv) // 10)
                self.sld_prefix.setPageStep(page)
                self.sld_prefix.setSingleStep(1)
                self.sld_prefix.setTickInterval(page)
                self.sld_prefix.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                self.sld_prefix.setValue(self.current_frame)
                self.sld_prefix.blockSignals(False)
                self._update_prefix_label()
            else:
                arr = self.trajs[self.cur_tid] if self.cur_tid is not None else np.zeros((7,2))
                minv = 6
                maxv = max(minv + 1, len(arr) - 1)
                self.sld_prefix.blockSignals(True)
                self.sld_prefix.setRange(minv, maxv)
                page = max(1, (maxv - minv) // 10)
                self.sld_prefix.setPageStep(page)
                self.sld_prefix.setSingleStep(1)
                self.sld_prefix.setTickInterval(page)
                self.sld_prefix.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
                self.sld_prefix.setValue(max(minv, min(self.cur_idx, maxv)))
                self.sld_prefix.blockSignals(False)
                self._update_prefix_label()

        def _update_prefix_label(self):
            if self.use_frame_mode:
                ts = self.frame_list[self.current_frame] if self.frame_list else None
                self.lbl_prefix.setText(f"frame {self.current_frame+1}/{len(self.frame_list)}  ts={ts}")
            else:
                total = len(self.trajs[self.cur_tid]) if self.cur_tid is not None else 0
                self.lbl_prefix.setText(f"{self.cur_idx}/{total}")

        def _sync_prefix_slider(self):
            self.sld_prefix.blockSignals(True)
            if self.use_frame_mode:
                self.sld_prefix.setValue(self.current_frame)
            else:
                self.sld_prefix.setValue(self.cur_idx)
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

        def _on_prefix_slider_changed(self, val: int):
            if self.use_frame_mode:
                self.current_frame = val
            else:
                arr = self.trajs[self.cur_tid]
                self.cur_idx = max(6, min(val, len(arr) - 1))
            self._update_prefix_label()
            self.update_all()

        # ---------- Tuning handlers ----------
        def _update_sens(self):
            s_baf = self.sld_baf.value() / 10.0
            s_stat = self.sld_stat.value() / 10.0
            s_bbox = self.sld_bbox.value() / 10.0
            self.online_single.baf_sensitivity = s_baf; self.lbl_baf.setText(f"{s_baf:.2f}")
            self.online_single.stationary_sens = s_stat; self.lbl_stat.setText(f"{s_stat:.2f}")
            self.online_single.bbox_sens = s_bbox; self.lbl_bbox.setText(f"{s_bbox:.2f}")
            # propagate to pool
            self.pool.cfg.update(baf_sensitivity=s_baf, stationary_sens=s_stat, bbox_sens=s_bbox)
            self.pool.reset()
            self.update_all()

        def _apply_tuning(self):
            self.online_single.rev_angle_deg = float(self.spn_rev.value())
            self.online_single.window = int(self.spn_win.value())
            self.online_single.enter_baf = int(self.spn_enter.value())
            self.online_single.exit_baf = int(self.spn_exit.value())
            # propagate to pool and reset states (hysteresis depends on config)
            self.pool.cfg.update(window=self.online_single.window,
                                 enter_baf=self.online_single.enter_baf,
                                 exit_baf=self.online_single.exit_baf,
                                 rev_angle_deg=self.online_single.rev_angle_deg)
            self.pool.reset()
            self.update_all()

        # ---------- Playback ----------
        def change_tid(self, idx):
            if not self.tids: return
            self.cur_tid = self.tids[idx]
            self.cur_idx = 6
            self._setup_prefix_slider()
            self.update_all()

        def toggle(self):
            (self.timer.stop() if self.timer.isActive() else self.timer.start())
            self.btn_play.setText("Pause" if self.timer.isActive() else "Play")

        def step_forward(self):
            if self.use_frame_mode:
                self.current_frame = min(len(self.frame_list) - 1, self.current_frame + 1)
                self.update_all()
                if self.current_frame >= len(self.frame_list) - 1:
                    self.timer.stop(); self.btn_play.setText("Play")
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

        # ---------- Drawing helpers ----------
        def _draw_background(self):
            if self._bg_img is not None and self._view_extent is not None:
                self.ax_main.imshow(
                    self._bg_img,
                    extent=self._view_extent,
                    origin=self.bg_origin,
                    alpha=self.bg_alpha,
                    zorder=0
                )
            self.ax_main.set_aspect('equal', adjustable='datalim')
            try: self.ax_main.set_box_aspect(1)
            except Exception: pass
            if self._view_extent is not None:
                xmin, xmax, ymin, ymax = self._view_extent
                self.ax_main.set_xlim(xmin, xmax)
                self.ax_main.set_ylim(ymin, ymax)
                self.ax_main.autoscale(False)

        def draw_entries(self):
            colors = plt.cm.tab10.colors
            for cid, (center, w, h, ang) in enumerate(self.model.entry_ellipses):
                E = ellipse_poly(center, w, h, ang)
                self.ax_main.plot(E[:, 0], E[:, 1], color=colors[cid % 10], lw=2, alpha=0.9, zorder=2)
                self.ax_main.scatter([center[0]], [center[1]], c=[colors[cid % 10]], s=60,
                                     edgecolor='k', zorder=5)
                self.ax_main.text(center[0], center[1], f"ID {cid}", fontsize=9, ha='center', va='bottom',
                                  color='black', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", alpha=0.7),
                                  zorder=6)

        # ---------- Main update ----------
        def update_all(self):
            self.ax_main.clear()
            self._draw_background()

            if self.use_frame_mode:
                # List active objects this frame
                objs = self.frame_objects[self.current_frame]
                active_tids = sorted(objs.keys())

                # Populate object combo
                self.cb_objects.blockSignals(True)
                self.cb_objects.clear()
                for tid in active_tids:
                    self.cb_objects.addItem(str(tid))
                self.cb_objects.blockSignals(False)

                # Draw all active points
                for tid, (x, y) in objs.items():
                    self.ax_main.scatter([x],[y], s=70, label=f"ID {tid}", zorder=4)
                    self.ax_main.text(x, y, f"{tid}", fontsize=9, color='k',
                                      bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="k", alpha=0.6), zorder=5)

                self.draw_entries()

                # If nothing active, just refresh canvases and return
                if not active_tids:
                    self.ax_main.set_title(f"Frame {self.current_frame+1}/{len(self.frame_list)} — no active objects")
                    self.canvas_main.draw()
                    self.ax_route.clear(); self.canvas_route.draw()
                    self.ax_micro.clear(); self.canvas_micro.draw()
                    self.table.setRowCount(0)
                    self._sync_prefix_slider()
                    return

                # Selected object
                try:
                    sel_tid = int(self.cb_objects.currentText())
                except Exception:
                    sel_tid = active_tids[0]

                # Update per-object online state up to this frame
                # (replays if needed to keep hysteresis)
                online_sel, prefix_sel = self.pool.update_to_frame(sel_tid, self.tid_frame_history.get(sel_tid, []), self.current_frame)
                # Draw selected object's trajectory prefix (thin)
                if prefix_sel is not None:
                    self.ax_main.plot(prefix_sel[:,0], prefix_sel[:,1], color='tab:blue', lw=2, zorder=3)
                    self.ax_main.scatter(prefix_sel[-1,0], prefix_sel[-1,1], c='yellow', edgecolor='k', s=60, zorder=6, label='Selected current')

                # Route & micro panels for selected object
                if prefix_sel is not None and len(prefix_sel) >= 6:
                    # predict top-k
                    classes, probs, unc = self.model.predict_segment_ml(prefix_sel)
                    labels = [f"{a}->{b}" for (a, b) in classes]
                    self.ax_route.clear()
                    self.ax_route.barh(labels[::-1], probs[::-1], color='tab:green')
                    self.ax_route.set_xlim(0, 1)
                    self.ax_route.set_title(f"Route (unc={unc:.2f})")
                    self.canvas_route.draw()

                    # micro
                    info = online_sel.update(prefix_sel)
                    mp = info["mode_probs"]
                    self.ax_micro.clear()
                    self.ax_micro.barh(["moving", "loading", "stationary"],
                                       [mp["moving"], mp["loading"], mp["stationary"]],
                                       color=['tab:gray', 'tab:orange', 'tab:green'])
                    self.ax_micro.set_xlim(0, 1)
                    self.ax_micro.set_title(f"Maneuver = {info['mode']}")
                    self.canvas_micro.draw()

                    # Heatmap overlay for selected object
                    if self.chk_overlay.isChecked():
                        _, parts = forecast_particles(
                            prefix_sel, online_sel,
                            horizon=5,
                            n_particles=1200,
                            seed=42,
                            use_progress_dyn=self.use_progress_dyn
                        )
                        draw_heatmap_and_ci95_on_ax(self.ax_main, parts, label="CI95 (H=1)")
                    self.ax_main.set_title(
                        f"Frame {self.current_frame+1}/{len(self.frame_list)} — Selected TID {sel_tid} "
                        f"{info['display_label']} (conf={info['display_confidence']:.2f})"
                    )
                else:
                    self.ax_route.clear(); self.ax_route.set_title("Route (insufficient prefix)"); self.canvas_route.draw()
                    self.ax_micro.clear(); self.ax_micro.set_title("Maneuver (insufficient prefix)"); self.canvas_micro.draw()
                    self.ax_main.set_title(
                        f"Frame {self.current_frame+1}/{len(self.frame_list)} — Selected TID {sel_tid} (insufficient prefix)"
                    )

                # Table: predictions for all active objects
                rows = []
                for tid in active_tids:
                    online_i, prefix_i = self.pool.update_to_frame(tid, self.tid_frame_history.get(tid, []), self.current_frame)
                    if prefix_i is None or len(prefix_i) < 6:
                        rows.append((tid, "-", np.nan, "-", np.nan, np.nan, np.nan))
                        continue
                    info_i = online_i.update(prefix_i)
                    rows.append((tid,
                                 info_i["display_label"],
                                 info_i["display_confidence"],
                                 info_i["mode"],
                                 info_i["mode_probs"]["moving"],
                                 info_i["mode_probs"]["loading"],
                                 info_i["mode_probs"]["stationary"]))
                self.table.setRowCount(len(rows))
                for r, row in enumerate(rows):
                    for c, val in enumerate(row):
                        item = QtWidgets.QTableWidgetItem("" if (isinstance(val, float) and np.isnan(val)) else str(val))
                        self.table.setItem(r, c, item)

                self.ax_main.set_xlabel("X"); self.ax_main.set_ylabel("Y")
                self.ax_main.legend(loc='best')
                self.canvas_main.draw()
                self._sync_prefix_slider()
                return

            # ------- Trajectory mode (original) -------
            if not self.tids:
                self.ax_main.set_title("No trajectories loaded")
                self.canvas_main.draw()
                return

            arr = self.trajs[self.cur_tid]
            if self.cur_idx <= 6: self.cur_idx = 6
            seg = arr[:self.cur_idx]

            # Draw trajectory
            self.ax_main.plot(arr[:, 0], arr[:, 1], color='lightgray', lw=0.8, alpha=0.6,
                              label=f"Traj {self.cur_tid}", zorder=1)
            self.ax_main.plot(seg[:, 0], seg[:, 1], color='tab:blue', lw=2, zorder=3)
            self.ax_main.scatter(seg[-1, 0], seg[-1, 1], c='yellow', edgecolor='k', s=60,
                                 zorder=5, label='Current')
            self.draw_entries()

            # Predictions
            info = self.online_single.update(seg)

            if self.chk_overlay.isChecked():
                _, parts = forecast_particles(
                    seg, self.online_single,
                    horizon=5,
                    n_particles=1200,
                    seed=42,
                    use_progress_dyn=self.use_progress_dyn
                )
                draw_heatmap_and_ci95_on_ax(self.ax_main, parts, label="CI95 (H=1)")

            self.ax_main.set_title(
                f"Trajectory {self.cur_tid} [{self.cur_idx}/{len(arr)}] "
                f"{info['display_label']} (conf={info['display_confidence']:.2f})"
            )
            self.ax_main.set_xlabel("X")
            self.ax_main.set_ylabel("Y")
            self.ax_main.legend(loc='best')
            self.canvas_main.draw()

            # Route panel
            classes, probs, unc = self.model.predict_segment_ml(seg)
            labels = [f"{a}->{b}" for (a, b) in classes]
            self.ax_route.clear()
            self.ax_route.barh(labels[::-1], probs[::-1], color='tab:green')
            self.ax_route.set_xlim(0, 1)
            self.ax_route.set_title(f"Route (unc={unc:.2f})")
            self.canvas_route.draw()

            # Micro panel
            mp = info["mode_probs"]
            self.ax_micro.clear()
            self.ax_micro.barh(["moving", "loading", "stationary"],
                               [mp["moving"], mp["loading"], mp["stationary"]],
                               color=['tab:gray', 'tab:orange', 'tab:green'])
            self.ax_micro.set_xlim(0, 1)
            self.ax_micro.set_title(f"Maneuver = {info['mode']}")
            self.canvas_micro.draw()

            self._sync_prefix_slider()

        # ---------- Heatmaps dialog ----------
        def show_heatmaps(self):
            from PyQt6 import QtWidgets
            if self.use_frame_mode:
                # Selected object
                try:
                    sel_tid = int(self.cb_objects.currentText())
                except Exception:
                    QtWidgets.QMessageBox.information(self, "No object", "No active object selected.")
                    return
                online_sel, prefix_sel = self.pool.update_to_frame(sel_tid, self.tid_frame_history.get(sel_tid, []), self.current_frame)
                if prefix_sel is None or len(prefix_sel) < 6:
                    QtWidgets.QMessageBox.information(self, "Insufficient prefix", "Not enough history to plot heatmaps.")
                    return
                info1, p1 = forecast_particles(prefix_sel, online_sel, horizon=1, n_particles=1500, seed=1, use_progress_dyn=self.use_progress_dyn)
                info2, p2 = forecast_particles(prefix_sel, online_sel, horizon=2, n_particles=1500, seed=2, use_progress_dyn=self.use_progress_dyn)
                info3, p3 = forecast_particles(prefix_sel, online_sel, horizon=3, n_particles=1500, seed=3, use_progress_dyn=self.use_progress_dyn)
            else:
                arr = self.trajs[self.cur_tid]
                seg = arr[:self.cur_idx]
                info1, p1 = forecast_particles(seg, self.online_single, horizon=1, n_particles=1500, seed=1, use_progress_dyn=self.use_progress_dyn)
                info2, p2 = forecast_particles(seg, self.online_single, horizon=2, n_particles=1500, seed=2, use_progress_dyn=self.use_progress_dyn)
                info3, p3 = forecast_particles(seg, self.online_single, horizon=3, n_particles=1500, seed=3, use_progress_dyn=self.use_progress_dyn)

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Future position heatmaps (H=1,2,3)")
            lay = QtWidgets.QVBoxLayout(dlg)
            fig = Figure(figsize=(10, 3), tight_layout=True)
            axs = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
            for H, ax, parts in zip([1,2,3], axs, [p1,p2,p3]):
                H2, xe, ye = np.histogram2d(parts[:,0], parts[:,1], bins=100)
                ax.pcolormesh(xe, ye, H2.T, cmap="magma", shading="auto")
                ax.scatter([parts.mean(axis=0)[0]],[parts.mean(axis=0)[1]], c='cyan', s=25)
                ax.set_title(f"H={H}")
            lay.addWidget(FigureCanvas(fig))
            dlg.exec()

        # ---------- Export predictions to JSON ----------
        def _compute_predictions_all_frames(self) -> list:
            """
            Returns a list aligned with frames:
              [
                { "time_stamp": ..., "predictions": [
                     { "track_id": tid, "route": {"from": a, "to": b, "prob": p, "uncertainty": u},
                       "mode": m, "mode_probs": {"moving":..., "loading":..., "stationary":...} }, ...
                  ] },
                ...
              ]
            """
            if not self.use_frame_mode:
                return []

            # reset pool to ensure a clean deterministic pass
            self.pool.reset()
            out = []
            for fi, ts in enumerate(self.frame_list):
                preds = []
                objs = self.frame_objects[fi]
                for tid in sorted(objs.keys()):
                    online_i, prefix_i = self.pool.update_to_frame(tid, self.tid_frame_history.get(tid, []), fi)
                    if prefix_i is None or len(prefix_i) < 6:
                        continue
                    info_i = online_i.update(prefix_i)
                    a, b = info_i["route_pair"]
                    preds.append({
                        "track_id": int(tid),
                        "route": {"from": int(a), "to": int(b),
                                  "prob": float(info_i["route_prob"]),
                                  "uncertainty": float(info_i["route_uncertainty"])},
                        "mode": info_i["mode"],
                        "mode_probs": {k: float(v) for k, v in info_i["mode_probs"].items()},
                        "display_label": info_i["display_label"],
                        "display_confidence": float(info_i["display_confidence"])
                    })
                out.append({"time_stamp": ts, "predictions": preds})
            return out

        def save_predictions_dialog(self):
            from PyQt6 import QtWidgets
            if not self.use_frame_mode:
                QtWidgets.QMessageBox.information(self, "Frame-mode only", "Export is only available in frame-mode.")
                return
            # Compute predictions
            preds = self._compute_predictions_all_frames()
            if len(preds)==0:
                QtWidgets.QMessageBox.information(self, "No predictions", "No predictions to save.")
                return
            # Choose path (default near original file)
            default_name = "predictions_" + (os.path.basename(self.source_json_path) if self.source_json_path else "frames.json")
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save predictions", default_name, "JSON (*.json)"
            )
            if not out_path:
                return
            try:
                with open(out_path, "w") as f:
                    json.dump(preds, f, indent=2)
                QtWidgets.QMessageBox.information(self, "Saved", f"Predictions saved to:\n{out_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Could not save predictions:\n{e}")

    # Launch window
    w = TrajPlayer(trajs, model, online,
                   frame_list=frame_list,
                   frame_objects=frame_objects,
                   tid_frame_history=tid_frame_history,
                   orig_frames=orig_frames,
                   source_json_path=source_json_path)
    w.resize(1280, 900); w.show()
    sys.exit(app.exec())

# ===== Main (CLI) =====
def main():
    parser = argparse.ArgumentParser(description="Trajectory ML + Micro Maneuvers + GUI (frame-aware)")
    parser.add_argument("--data", type=str, default="JSON-data/final.json", help="Path to trajectories JSON")    
    parser.add_argument("--entries", type=int, default=5, help="K (entry areas)")
    parser.add_argument("--codebook", type=int, default=16, help="K (step codebook)")
    parser.add_argument("--pca", type=int, default=6, help="PCA components for shape")
    parser.add_argument("--win", type=int, default=25, help="Window size for micro mode")
    parser.add_argument("--progress-dyn", action="store_true", help="Use class progress-aware dynamics")
    parser.add_argument("--demo", action="store_true", help="Train & forecast demo (no GUI)")
    parser.add_argument("--report", action="store_true", help="Create accuracy vs prefix report PNG")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (load or train)")
    parser.add_argument("--export-pred", type=str, default=None, help="Export predictions to JSON (frame-based input only, headless)")
    args = parser.parse_args()

    # Try to detect format
    with open(args.data, "r") as f:
        head = json.load(f)

    is_frame = isinstance(head, list)
    if is_frame:
        # Build training trajs from frame data too
        frame_list, frame_objects, tid_frame_history, raw_frames = load_frame_based(args.data)
        # Convert sparse histories into pure XY array for training fallback
        trajs_for_train = {}
        for tid, hist in tid_frame_history.items():
            pts = [xy for (fi, xy) in hist]
            if len(pts) >= 3:
                trajs_for_train[tid] = np.asarray(pts, dtype=float)
        if not trajs_for_train:
            print("Warning: No sufficient trajectories extracted from frame-based file; training may not work.")
        trajs = trajs_for_train
    else:
        frame_list = frame_objects = tid_frame_history = raw_frames = None
        trajs = load_trajectories(args.data)

    print(f"Loaded {'frame-based' if is_frame else 'trajectory-based'} data from {args.data}")

    if args.report:
        saved = list_saved_models()
        if saved:
            model = MLModel.load(saved[0]); print(f"Loaded model: {saved[0]}")
        else:
            model = auto_tune_and_train_ml(trajs, K_ENTRIES=args.entries, K_CODE=args.codebook, N_PCA=args.pca)
            path = timestamped_model_path(); model.save(path); print(f"Trained and saved: {path}")
        res = evaluate_prefix_accuracy(trajs, model)
        out = os.path.join(models_dir(), f"report_acc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        save_accuracy_plot(res, out); print(f"Saved accuracy report: {out}")
        return

    # Train or load a model for demo/export
    if is_frame and args.export_pred:
        # Headless export predictions
        # Train or load a model
        saved = list_saved_models()
        if saved:
            model = MLModel.load(saved[0]); print(f"Loaded model: {saved[0]}")
        else:
            model = auto_tune_and_train_ml(trajs, K_ENTRIES=args.entries, K_CODE=args.codebook, N_PCA=args.pca)
            path = timestamped_model_path(); model.save(path); print(f"Trained and saved: {path}")

        # Online pool
        T = auto_tune_mode_thresholds(trajs, win=args.win, rev_angle_deg=120.0)
        pool = OnlinePool(model, T, window=args.win, enter_baf=3, exit_baf=5)

        # Full pass
        out = []
        pool.reset()
        for fi, ts in enumerate(frame_list):
            preds = []
            objs = frame_objects[fi]
            for tid in sorted(objs.keys()):
                online_i, prefix_i = pool.update_to_frame(tid, tid_frame_history.get(tid, []), fi)
                if prefix_i is None or len(prefix_i) < 6:
                    continue
                info_i = online_i.update(prefix_i)
                a, b = info_i["route_pair"]
                preds.append({
                    "track_id": int(tid),
                    "route": {"from": int(a), "to": int(b),
                              "prob": float(info_i["route_prob"]),
                              "uncertainty": float(info_i["route_uncertainty"])},
                    "mode": info_i["mode"],
                    "mode_probs": {k: float(v) for k, v in info_i["mode_probs"].items()},
                    "display_label": info_i["display_label"],
                    "display_confidence": float(info_i["display_confidence"])
                })
            out.append({"time_stamp": ts, "predictions": preds})

        with open(args.export_pred, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Predictions exported to: {args.export_pred}")
        return

    if args.demo:
        model = auto_tune_and_train_ml(trajs, K_ENTRIES=args.entries, K_CODE=args.codebook, N_PCA=args.pca)
        savep = timestamped_model_path(); model.save(savep); print(f"Saved model -> {savep}")
        # Online classifier default thresholds
        T = auto_tune_mode_thresholds(trajs, win=args.win, rev_angle_deg=120.0)
        online = OnlineClassifier(model, T, window=args.win, enter_baf=3, exit_baf=5)
        # Confidence growth
        if trajs:
            tid = sorted(trajs.keys())[0]; arr = trajs[tid]
            for frac in [0.15,0.35,0.55,0.75,0.95]:
                n=max(6,int(len(arr)*frac)); classes,probs,unc = model.predict_segment_ml(arr[:n])
                print(f"prefix={n:5d} top1={classes[0]} p={probs[0]:.3f} unc={unc:.3f}")
            # Save example heatmaps
            seg = arr[: min(max(200, len(arr)//3), len(arr)-1) ]
            for H in (1,2,3):
                info, parts = forecast_particles(seg, online, horizon=H, n_particles=3000, seed=H, use_progress_dyn=args.progress_dyn)
                H2, xe, ye = np.histogram2d(parts[:,0], parts[:,1], bins=100)
                fig, ax = plt.subplots(figsize=(6,5))
                pcm = ax.pcolormesh(xe, ye, H2.T, cmap="magma", shading="auto")
                ax.scatter([seg[-1,0]],[seg[-1,1]], c='yellow', edgecolor='k', s=60, zorder=5)
                ax.set_title(f"{info['display_label']} "
                             f"H={H} "
                             f"conf={info['display_confidence']:.2f} "
                             f"progress_dyn={args.progress_dyn}")
                fig.colorbar(pcm, ax=ax, label="Density")
                out = f"demo_forecast_H{H}.png"; fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
                print(f"Saved {out}")
        else:
            print("No trajectories available for demo.")
        return

    if args.gui:
        # If we already trained/loaded in --report/--demo skip; else train minimally if needed
        saved = list_saved_models()
        model = MLModel.load(saved[0]) if saved else auto_tune_and_train_ml(trajs, K_ENTRIES=args.entries, K_CODE=args.codebook, N_PCA=args.pca)
        if not saved:
            path = timestamped_model_path(); model.save(path); print(f"Trained and saved: {path}")
        T = auto_tune_mode_thresholds(trajs, win=args.win, rev_angle_deg=120.0)
        online = OnlineClassifier(model, T, window=args.win, enter_baf=3, exit_baf=5)
        run_gui(trajs, model, online,
                K_ENTRIES=args.entries, K_CODE=args.codebook, N_PCA=args.pca, WIN=args.win,
                use_progress_dyn=True,
                frame_list=frame_list, frame_objects=frame_objects,
                tid_frame_history=tid_frame_history,
                orig_frames=raw_frames,
                source_json_path=args.data)
        return

    print("Use --demo for a non-GUI run, --report for accuracy plot, --gui to launch the app, or --export-pred to export predictions (frame-based).")

if __name__ == "__main__":
    main()
