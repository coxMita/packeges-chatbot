"""Score a package and explain which features drove the score.

Deliberately imports `features.extract_features` -- the *same* function used to
build the training set. Reimplementing extraction for serving is the classic way
to ship a model whose inputs quietly drift from what it was trained on.

The output carries `evidence`: the top SHAP contributions, each with a
human-readable description from `FEATURE_DESCRIPTIONS` and a `training` block
saying how common the observed value was among the malicious and the benign
packages the model learned from. That list is what the LLM is given. It cannot
see the source, so it cannot form its own opinion about the package -- it can
only explain the model's.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from config import MODELS  # noqa: E402
from behaviour import capabilities  # noqa: E402
from features import describe, extract_features, feature_names  # noqa: E402

TOP_EVIDENCE = 8

# A feature at 0 that pushes the score down is not interesting to a reader
# ("it does NOT call subprocess" is noise). We report contributions above this.
MIN_ABS_SHAP = 1e-3


class TrainingReference:
    """Per-class distribution of every feature over the real training packages
    (written by ml/train_gbdt.py), to say how the observed value compares."""

    def __init__(self, path: Path) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as z:
                self.arrays = {k: z[k] for k in z.files}

    def context(self, feature: str, value: float) -> dict[str, Any] | None:
        mal = self.arrays.get(f"malicious/{feature}")
        ben = self.arrays.get(f"benign/{feature}")
        if mal is None or ben is None or not len(mal) or not len(ben):
            return None
        # Compare on the side of the distribution the value sits on: "at most 3
        # files" is the informative tail for a tiny package, "at least 12
        # subprocess calls" for a busy one.
        median = float(np.median(np.concatenate([mal, ben])))
        tail = "le" if value <= median else "ge"

        def share(a: np.ndarray) -> float:
            if tail == "le":
                return float(np.searchsorted(a, value, side="right") / len(a))
            return float(1 - np.searchsorted(a, value, side="left") / len(a))

        m, b = share(mal), share(ben)
        return {
            "tail": tail,
            "malicious_share": round(m, 4),
            "benign_share": round(b, 4),
            # How many times more common among malware; capped so a value never
            # seen in benign training data does not read as "infinitely" so.
            "ratio": round(min(m / max(b, 1e-3), 999.0), 2),
            "n_malicious": int(len(mal)),
            "n_benign": int(len(ben)),
        }


@dataclass
class Evidence:
    feature: str
    value: float
    contribution: float          # SHAP value: >0 pushes toward malicious
    description: str
    training: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": round(float(self.value), 4),
            "contribution": round(float(self.contribution), 4),
            "direction": "malicious" if self.contribution > 0 else "benign",
            "description": self.description,
            "training": self.training,
        }


@dataclass
class Verdict:
    verdict: str                 # "malicious" | "benign"
    confidence: float            # 0-1, confidence in the stated verdict
    malicious_probability: float
    threshold: float
    evidence: list[Evidence] = field(default_factory=list)
    model_agreement: str | None = None
    embed_probability: float | None = None
    # Concrete malicious capabilities found in the code (ml/behaviour.py).
    capabilities: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence * 100, 1),
            "malicious_probability": round(self.malicious_probability, 4),
            "threshold": round(self.threshold, 3),
            "model_agreement": self.model_agreement,
            "embed_probability": (
                round(self.embed_probability, 4)
                if self.embed_probability is not None else None
            ),
            "evidence": [e.to_dict() for e in self.evidence],
            "capabilities": self.capabilities,
        }


class Scorer:
    """Loads the trained artifacts once and scores packages on demand."""

    def __init__(self, models_dir: Path = MODELS):
        self.feature_names = feature_names()
        self.gbdt: dict | None = None
        self.explainer = None
        self.reference = TrainingReference(models_dir / "feature_reference.npz")

        gbdt_path = models_dir / "gbdt.pkl"
        if gbdt_path.exists():
            with open(gbdt_path, "rb") as fh:
                self.gbdt = pickle.load(fh)
            import shap
            self.explainer = shap.TreeExplainer(self.gbdt["model"])
            # Score with the columns the model was trained on, in its order;
            # the extractor may produce more than an older model knows.
            self.feature_names = list(self.gbdt["feature_names"])

    @property
    def ready(self) -> bool:
        return self.gbdt is not None

    def score(self, root: Path, package_name: str,
              popular_names: set[str] | None = None) -> Verdict:
        if not self.ready:
            raise RuntimeError(
                "no trained model found -- run ml/train_gbdt.py first"
            )

        feats = extract_features(root, package_name, popular_names or set())
        x = np.array([[feats[c] for c in self.feature_names]], dtype=np.float32)

        prob = float(self.gbdt["model"].predict(x)[0])
        threshold = float(self.gbdt["threshold"])
        is_malicious = prob >= threshold

        # Confidence is distance from the decision boundary, rescaled to 0-1 on
        # whichever side of the threshold we landed. Reporting the raw
        # probability as "confidence" would claim 51% certainty at the boundary.
        if is_malicious:
            confidence = (prob - threshold) / max(1.0 - threshold, 1e-9)
        else:
            confidence = (threshold - prob) / max(threshold, 1e-9)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return Verdict(
            verdict="malicious" if is_malicious else "benign",
            confidence=confidence,
            malicious_probability=prob,
            threshold=threshold,
            evidence=self._evidence(x, feats),
            capabilities=[{"feature": c, "count": round(float(feats[c]), 2), "description": describe(c)}
                          for c in capabilities(feats)],
        )

    def _evidence(self, x: np.ndarray, feats: dict[str, float]) -> list[Evidence]:
        """Top SHAP contributions, largest absolute effect first."""
        vals = self.explainer.shap_values(x)
        # LightGBM binary returns either a (1, n) array or a 2-element list.
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        vals = np.asarray(vals).reshape(-1)

        ranked = sorted(
            zip(self.feature_names, vals),
            key=lambda kv: -abs(kv[1]),
        )
        out = [
            Evidence(
                feature=name,
                value=feats[name],
                contribution=float(sv),
                description=describe(name),
                training=self.reference.context(name, feats[name]),
            )
            for name, sv in ranked
            if abs(sv) >= MIN_ABS_SHAP
        ]
        return out[:TOP_EVIDENCE]
