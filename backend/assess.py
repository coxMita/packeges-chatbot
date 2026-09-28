"""Combine the classifier and the similarity search into one assessment.

Two outputs, both built from ml/models/calibration.json (see ml/calibrate.py):

  * a tier -- malicious when both methods flag the package, or the classifier
    alone clears its strict threshold (0.1% out-of-fold false alarms);
    suspicious when one flags it or the classifier clears its lower review
    threshold (1%); clean otherwise. Suspicious is what a human should read.
  * the chance the package really is malware, as a range. The calibrator was
    fitted on the validation split, whose malware share (recorded as
    `base_rate`) is nowhere near real PyPI's, so the answer is re-weighted to
    two prior rates:
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

# Measured, not modelled (ml/report.py, live set): of 27 OSSF-reported malware
# packages absent from the dataset and still downloadable from PyPI -- malware
# that has so far evaded removal -- 7 reach malicious and 2 suspicious. Several
# of the rest show no payload in the version still published.
NOVEL_MALWARE_CAUGHT = "9 of 27"


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

    def assess(self, clf_prob: float, sim_share: float | None,
               has_capability: bool = True) -> dict | None:
        """`sim_share` is in [0, 1], or None when similarity was unavailable.

        `has_capability`: the code can do something concretely malicious
        (ml/behaviour.CAPABILITY). With the gate on, a package flagged on shape
        alone is held at suspicious rather than called malicious.
        """
        if not self.ready:
            return None
        cal = self.cal

        clf_flag = clf_prob >= cal["classifier_threshold"]
        # Older calibration files predate the review/strict thresholds.
        clf_review = clf_prob >= cal.get("review_threshold", cal["classifier_threshold"])
        clf_strict = clf_prob >= cal.get("strict_threshold", 1.1)
        if sim_share is None:
            model, x = cal["classifier_only"], [_logit(clf_prob)]
            sim_flag = False
            basis = "classifier only (similarity unavailable)"
        else:
            model, x = cal["both"], [_logit(clf_prob), sim_share]
            sim_flag = sim_share >= cal["sim_flag"]
            basis = "classifier and code similarity"
        if cal.get("capability_gate"):
            # Soft capability gate (ml/generalize.py): chosen on unseen malware
            # families, 2025 months, and held on 2026 months at ~95% precision.
            # Malicious needs the classifier AND a concrete capability, or the
            # classifier alone at its strict (0.1% false-alarm) threshold.
            tier = ("malicious" if (clf_flag and has_capability) or clf_strict
                    else "suspicious" if clf_flag or sim_flag or clf_review
                    else "clean")
            if clf_strict and not has_capability:
                basis += "; the classifier alone is confident enough, though no concrete capability was found"
            elif clf_flag and not has_capability:
                basis += ("; the score is high but no concrete malicious capability (code that "
                          "runs by itself, a payload, exfiltration) was found, so review it")
            elif clf_review and not clf_flag:
                basis += "; below the classifier's alarm threshold but high enough to review"
            elif sim_flag and not clf_flag:
                basis += "; the code resembles known malware, but the classifier does not flag it"
        else:
            # Without similarity there is no second vote to wait for.
            tier = ("malicious" if clf_strict or (clf_flag and (sim_flag or sim_share is None))
                    else "suspicious" if clf_flag or sim_flag or clf_review
                    else "clean")
            if clf_strict and not sim_flag:
                basis += "; the classifier alone is confident enough"
            elif clf_review and not clf_flag and not sim_flag:
                basis += "; below the classifier's alarm threshold but high enough to review"

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
            "has_capability": bool(has_capability),
        }
