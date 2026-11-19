# ==========================================================
# safety_framework.py
# Phase 1
# ==========================================================

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np



def route_prediction_uncertainty(
    route_info
):
    if route_info is None:
        return 1.0

    unc = route_info.get(
        "uncertainty"
    )

    if unc is not None:
        return float(
            np.clip(
                unc,
                0.0,
                1.0
            )
        )

    p = route_info.get(
        "prob"
    )

    if p is None:
        return 1.0

    return float(
        np.clip(
            1.0 - float(p),
            0.0,
            1.0
        )
    )

def maneuver_uncertainty(mode_info):
    if not mode_info:
        return 1.0

    mode = mode_info.get("mode")
    probabilities = mode_info.get("mode_probs", {})

    if mode is None or mode not in probabilities:
        return 1.0

    confidence = float(probabilities[mode])

    return float(np.clip(
        1.0 - confidence,
        0.0,
        1.0,
    ))

def estimate_bbox_uncertainty(
    track_id,
    bbox_model
):
    model = bbox_model.get(
        track_id
    )

    if model is None:
        return 0.0

    eigvals = np.linalg.eigvalsh(
        model["cov"]
    )

    raw = np.sqrt(
        max(
            0.0,
            eigvals.max()
        )
    )

    score = raw / 10.0

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )

def estimate_trajectory_uncertainty(
    track_id,
    track_history
):
    hist = track_history.get(
        track_id,
        []
    )

    if len(hist) < 6:
        return 0.0

    pts = np.asarray([
        (h[1], h[2])
        for h in hist[-10:]
    ])

    vel = np.diff(
        pts,
        axis=0
    )

    if len(vel) < 2:
        return 0.0

    headings = np.arctan2(
        vel[:,1],
        vel[:,0]
    )

    return float(
        np.clip(
            np.std(headings)
            / np.pi,
            0.0,
            1.0
        )
    )



# ==========================================================
# RAW SCENE DESCRIPTION
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

    extra: dict = field(
        default_factory=dict
    )


# ==========================================================
# SUPERVISOR OUTPUT
# ==========================================================

@dataclass
class SupervisorResult:

    name: str

    category: str

    score: float

    confidence: float

    flag: bool = False

    enabled: bool = True

    details: dict = field(
        default_factory=dict
    )


# ==========================================================
# ASSESSMENT
# ==========================================================

@dataclass
class TrustAssessment:

    trust_score: float

    runtime_state: str

    supervisor_results: list


# ==========================================================
# BASE CLASS
# ==========================================================

class Supervisor(ABC):

    name = "base"

    category = "generic"

    enabled = True

    @abstractmethod
    def evaluate(
        self,
        scene: SceneContext
    ):
        pass


# ==========================================================
# MANAGER
# ==========================================================

class SupervisorManager:

    def __init__(self):

        self.supervisors = {}

    def register(
        self,
        supervisor
    ):

        self.supervisors[
            supervisor.name
        ] = supervisor

    def unregister(
        self,
        name
    ):

        self.supervisors.pop(
            name,
            None
        )

    def evaluate(
        self,
        scene
    ):

        results = []

        for supervisor in self.supervisors.values():

            if not supervisor.enabled:
                continue

            try:

                result = supervisor.evaluate(
                    scene
                )

            except Exception as e:

                print(
                    f"[SUPERVISOR ERROR] "
                    f"{supervisor.name}: {e}"
                )

                result = SupervisorResult(
                    name=supervisor.name,
                    category="error",
                    score=1.0,
                    confidence=0.0,
                    flag=True,
                    details={
                        "exception": str(e)
                    }
                )

            results.append(
                result
            )

        return results

# ==========================================================
# DECISION ENGINE
# Phase 3
# ==========================================================

class DecisionEngine:

    DEFAULT_WEIGHTS = {

        "input_anomaly": 0.25,

        "uncertainty": 0.15,

        "surrogate": 0.20,

        "output_anomaly": 0.15,

        "model_anomaly": 0.25
    }

    def __init__(
        self,
        weights=None
    ):

        self.weights = (
            weights
            if weights is not None
            else self.DEFAULT_WEIGHTS.copy()
        )

    def compute_trust(
        self,
        results
    ):

        penalties = {
            r.name: r.score
            for r in results
        }

        trust = 1.0

        trust -= 0.30 * penalties.get(
            "camera_ood",
            0.0
        )

        trust -= 0.20 * penalties.get(
            "bbox_uncertainty",
            0.0
        )

        trust -= 0.20 * penalties.get(
            "trajectory_uncertainty",
            0.0
        )

        trust -= 0.15 * penalties.get(
            "track_uncertainty",
            0.0
        )

        trust -= 0.10 * penalties.get(
            "route_uncertainty",
            0.0
        )

        trust -= 0.05 * penalties.get(
            "maneuver_uncertainty",
            0.0
        )

        return float(
            np.clip(
                trust,
                0.0,
                1.0
            )
        )

class BBoxUncertaintySupervisor(
    Supervisor
):

    name = "bbox_uncertainty"

    category = "uncertainty"

    def evaluate(
        self,
        scene
    ):

        if (
            scene.active_track_id
            is None
        ):
            score = 0.0

        else:

            score = (
                estimate_bbox_uncertainty(
                    scene.active_track_id,
                    scene.extra.get(
                        "track_bbox_model",
                        {}
                    ),
                )
            )

        return SupervisorResult(
            name=self.name,
            category=self.category,
            score=score,
            confidence=1.0 - score,
            flag=(score > 0.7)
        )

class TrajectoryUncertaintySupervisor(
    Supervisor
):

    name = "trajectory_uncertainty"

    category = "uncertainty"

    def evaluate(
        self,
        scene
    ):

        if (
            scene.active_track_id
            is None
        ):
            score = 0.0

        else:

            score = (
                estimate_trajectory_uncertainty(
                    scene.active_track_id,
                    scene.extra.get(
                        "track_history",
                        {}
                    ),
                )
            )

        return SupervisorResult(
            name=self.name,
            category=self.category,
            score=score,
            confidence=1.0 - score,
            flag=(score > 0.7)
        )

class RouteUncertaintySupervisor(
    Supervisor
):

    name = "route_uncertainty"

    category = "uncertainty"

    def evaluate(
        self,
        scene
    ):

        score = (
            route_prediction_uncertainty(
                scene.route_prediction
            )
        )

        return SupervisorResult(

            name=self.name,

            category=self.category,

            score=score,

            confidence=1.0-score,

            flag=(
                score > 0.7
            )
        )

class ManeuverUncertaintySupervisor(
    Supervisor
):

    name = "maneuver_uncertainty"

    category = "uncertainty"

    def evaluate(
        self,
        scene
    ):

        score = (
            maneuver_uncertainty(
                scene.maneuver_prediction
            )
        )

        return SupervisorResult(

            name=self.name,

            category=self.category,

            score=score,

            confidence=1.0-score,

            flag=(
                score > 0.7
            )
        )

class SurrogateSupervisor(
    Supervisor
):

    name = "surrogate"

    category = "surrogate"

    def evaluate(
        self,
        scene
    ):

        return SupervisorResult(

            name=self.name,

            category=self.category,

            score=0.0,

            confidence=0.0,

            flag=False,

            details={
                "status":
                    "not_implemented"
            }
        )
