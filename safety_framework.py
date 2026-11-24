# ==========================================================
# safety_framework.py
# ==========================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ==========================================================
# Uncertainty estimators
# ==========================================================
def route_prediction_uncertainty(route_info):
    if route_info is None:
        return 1.0
    unc = route_info.get("uncertainty")
    if unc is not None:
        return float(np.clip(unc, 0.0, 1.0))
    p = route_info.get("prob")
    if p is None:
        return 1.0
    return float(np.clip(1.0 - float(p), 0.0, 1.0))


def maneuver_uncertainty(mode_info):
    if not mode_info:
        return 1.0
    mode = mode_info.get("mode")
    probs = mode_info.get("mode_probs", {})
    if mode is None or mode not in probs:
        return 1.0
    return float(np.clip(1.0 - float(probs[mode]), 0.0, 1.0))


def estimate_bbox_uncertainty(track_id, bbox_model):
    model = bbox_model.get(track_id)
    if model is None:
        return 0.0
    eigvals = np.linalg.eigvalsh(model["cov"])
    return float(np.clip(np.sqrt(max(0.0, eigvals.max())) / 10.0, 0.0, 1.0))


def estimate_trajectory_uncertainty(track_id, track_history):
    hist = track_history.get(track_id, [])
    if len(hist) < 6:
        return 0.0
    pts = np.asarray([(h[1], h[2]) for h in hist[-10:]])
    vel = np.diff(pts, axis=0)
    if len(vel) < 2:
        return 0.0
    headings = np.arctan2(vel[:, 1], vel[:, 0])
    return float(np.clip(np.std(headings) / np.pi, 0.0, 1.0))


# ==========================================================
# Scene description / result containers
# ==========================================================
@dataclass
class SceneContext:
    timestamp: float
    frame_idx: int
    active_track_id: int | None
    frame: object | None
    route_prediction: dict | None
    maneuver_prediction: dict | None
    ttc: float
    extra: dict = field(default_factory=dict)


@dataclass
class SupervisorResult:
    name: str
    category: str
    score: float
    confidence: float
    flag: bool = False
    enabled: bool = True
    details: dict = field(default_factory=dict)


@dataclass
class TrustAssessment:
    trust_score: float
    runtime_state: str
    supervisor_results: list


# ==========================================================
# Supervisor base + manager
# ==========================================================
class Supervisor(ABC):
    name = "base"
    category = "generic"
    enabled = True

    @abstractmethod
    def evaluate(self, scene: SceneContext):
        pass


class SupervisorManager:
    def __init__(self):
        self.supervisors = {}

    def register(self, supervisor):
        self.supervisors[supervisor.name] = supervisor

    def unregister(self, name):
        self.supervisors.pop(name, None)

    def evaluate(self, scene):
        results = []
        for s in self.supervisors.values():
            if not s.enabled:
                continue
            try:
                result = s.evaluate(scene)
            except Exception as e:
                print(f"[SUPERVISOR ERROR] {s.name}: {e}")
                result = SupervisorResult(name=s.name, category="error", score=1.0,
                                          confidence=0.0, flag=True,
                                          details={"exception": str(e)})
            results.append(result)
        return results


# ==========================================================
# Decision engine
# NOTE: compute_trust uses fixed per-supervisor penalties;
#       self.weights is currently not applied.
# ==========================================================
class DecisionEngine:
    DEFAULT_WEIGHTS = {
        "input_anomaly": 0.25,
        "uncertainty": 0.15,
        "surrogate": 0.20,
        "output_anomaly": 0.15,
        "model_anomaly": 0.25,
    }

    def __init__(self, weights=None):
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()

    def compute_trust(self, results):
        p = {r.name: r.score for r in results}
        trust = 1.0
        trust -= 0.30 * p.get("camera_ood", 0.0)
        trust -= 0.10 * p.get("bbox_uncertainty", 0.0)
        trust -= 0.20 * p.get("trajectory_uncertainty", 0.0)
        trust -= 0.15 * p.get("track_uncertainty", 0.0)
        trust -= 0.10 * p.get("route_uncertainty", 0.0)
        trust -= 0.05 * p.get("maneuver_uncertainty", 0.0)
        trust -= 0.10 * p.get("safety_margin", 0.0)
        trust -= 0.10 * p.get("plausibility", 0.0)
        return float(np.clip(trust, 0.0, 1.0))


# ==========================================================
# Supervisors
# ==========================================================
class BBoxUncertaintySupervisor(Supervisor):
    name, category = "bbox_uncertainty", "uncertainty"

    def evaluate(self, scene):
        score = 0.0
        if scene.active_track_id is not None:
            score = estimate_bbox_uncertainty(
                scene.active_track_id, scene.extra.get("track_bbox_model", {}))
        return SupervisorResult(self.name, self.category, score, 1.0 - score,
                                flag=(score > 0.7))


class TrajectoryUncertaintySupervisor(Supervisor):
    name, category = "trajectory_uncertainty", "uncertainty"

    def evaluate(self, scene):
        score = 0.0
        if scene.active_track_id is not None:
            score = estimate_trajectory_uncertainty(
                scene.active_track_id, scene.extra.get("track_history", {}))
        return SupervisorResult(self.name, self.category, score, 1.0 - score,
                                flag=(score > 0.7))


class RouteUncertaintySupervisor(Supervisor):
    name, category = "route_uncertainty", "uncertainty"

    def evaluate(self, scene):
        score = route_prediction_uncertainty(scene.route_prediction)
        return SupervisorResult(self.name, self.category, score, 1.0 - score,
                                flag=(score > 0.7))


class ManeuverUncertaintySupervisor(Supervisor):
    name, category = "maneuver_uncertainty", "uncertainty"

    def evaluate(self, scene):
        score = maneuver_uncertainty(scene.maneuver_prediction)
        return SupervisorResult(self.name, self.category, score, 1.0 - score,
                                flag=(score > 0.7))


class SurrogateSupervisor(Supervisor):
    name, category = "surrogate", "surrogate"

    def evaluate(self, scene):
        return SupervisorResult(self.name, self.category, 0.0, 0.0, flag=False,
                                details={"status": "not_implemented"})


class SafetyMarginSupervisor(Supervisor):
    name, category = "safety_margin", "uncertainty"

    def evaluate(self, scene):
        ttc = scene.ttc
        if not np.isfinite(ttc):
            return SupervisorResult(self.name, self.category, 0.0, 1.0)
        score = float(np.clip((2.0 - ttc) / 2.0, 0.0, 1.0))
        return SupervisorResult(self.name, self.category, score, 1.0 - score,
                                flag=(score > 0.5))


class PlausibilitySupervisor(Supervisor):
    name, category = "plausibility", "uncertainty"

    def evaluate(self, scene):
        viewer = scene.extra.get("viewer")
        topology_conf = 1.0
        if viewer is not None:
            topology_conf = float((viewer.topology_info or {}).get("confidence", 1.0))
        score = max(route_prediction_uncertainty(scene.route_prediction),
                    maneuver_uncertainty(scene.maneuver_prediction),
                    1.0 - topology_conf)
        return SupervisorResult(self.name, self.category, score, 1.0 - score,
                                flag=(score > 0.5))
