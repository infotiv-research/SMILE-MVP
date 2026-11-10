# ==========================================================
# safety_framework.py
# Phase 1
# ==========================================================

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


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

        trust = 1.0

        for result in results:

            w = self.weights.get(
                result.category,
                0.1
            )

            trust -= (
                w *
                result.score
            )

            if result.flag:
                trust -= 0.05

        return max(
            0.0,
            min(
                1.0,
                trust
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

            score = estimate_bbox_uncertainty(
                scene.active_track_id
            )

        return SupervisorResult(

            name=self.name,

            category=self.category,

            score=score,

            confidence=1.0 - score,

            flag=(
                score > 0.7
            )
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
                    scene.active_track_id
                )
            )

        return SupervisorResult(

            name=self.name,

            category=self.category,

            score=score,

            confidence=1.0 - score,

            flag=(
                score > 0.7
            )
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