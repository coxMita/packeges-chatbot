"""Combine the classifier and the similarity search into one assessment.

Two outputs, both built from ml/models/calibration.json (see ml/calibrate.py):

  * a tier -- malicious only when both methods flag the package, suspicious
    when exactly one does. On the held-out split the "malicious" tier had no
    false alarms at all; the disagreements are what a human should look at.
  * the chance the package really is malware, as a range. The calibrator was
    fitted on a test set that is ~26% malware; real PyPI is nowhere near that,
    so the answer is re-weighted to two prior rates:
        PRIOR_LOW   a package picked at random from PyPI
        PRIOR_HIGH  a package someone already had a reason to doubt
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from config import MODELS  # noqa: E402

PRIOR_LOW = 0.01
PRIOR_HIGH = 0.10
# "No threat found" next to a 25-79% chance of malware is a contradiction. When
# neither method crosses its own threshold but the combined, calibrated chance
# for an already-suspected package is at least this, the tier is raised to
# suspicious. On the held-out split: +2 malware and +2 benign moved to review.
SUSPICIOUS_CHANCE = 50.0
EPS = 1e-4

# Measured, not modelled: the calibrator only knows malware that resembles the
# training data. Of 7 malware packages reported to OSSF after the training
# snapshot, 2 reach the suspicious tier (websetup, pullgetsage); 5 read clean.
NOVEL_MALWARE_CAUGHT = "2 of 7"


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _reweight(p: float, base: float, prior: float) -> float:
    """Bayes on the odds: move a probability from one base rate to another."""
    odds = p / max(1 - p, EPS) * (prior / (1 - prior)) / (base / (1 - base))
    return odds / (1 + odds)


class Assessor:
    def __init__(self, path: Path = MODELS / "calibration.json") -> None:
        self.ready = False
        try:
            self.cal = json.loads(path.read_text())
            self.ready = True
        except (OSError, ValueError) as exc:
            print(f"[assess] no calibration ({exc}); run `python ml/calibrate.py`")

    def assess(self, clf_prob: float, sim_share: float | None) -> dict | None:
        """`sim_share` is in [0, 1], or None when similarity was unavailable."""
        if not self.ready:
            return None
        cal = self.cal

        clf_flag = clf_prob >= cal["classifier_threshold"]
        if sim_share is None:
            model, x = cal["classifier_only"], [_logit(clf_prob)]
            tier = "malicious" if clf_flag else "clean"
            basis = "classifier only (similarity unavailable)"
        else:
            model, x = cal["both"], [_logit(clf_prob), sim_share]
            sim_flag = sim_share >= cal["sim_flag"]
            tier = ("malicious" if clf_flag and sim_flag
                    else "suspicious" if clf_flag or sim_flag
                    else "clean")
            basis = "classifier and code similarity"

        z = model["intercept"] + sum(c * v for c, v in zip(model["coef"], x))
        p_test = 1 / (1 + math.exp(-z))
        chance_high = 100 * _reweight(p_test, cal["base_rate"], PRIOR_HIGH)
        if tier == "clean" and chance_high >= SUSPICIOUS_CHANCE:
            tier = "suspicious"
            basis += "; neither flags it alone, but together they lean malicious"

        return {
            "tier": tier,
            "classifier_flags": bool(clf_flag),
            "similarity_flags": None if sim_share is None else bool(sim_share >= cal["sim_flag"]),
            "chance_low": round(100 * _reweight(p_test, cal["base_rate"], PRIOR_LOW), 1),
            "chance_high": round(chance_high, 1),
            "prior_low": PRIOR_LOW,
            "prior_high": PRIOR_HIGH,
            "basis": basis,
            "novel_malware_caught": NOVEL_MALWARE_CAUGHT,
        }
