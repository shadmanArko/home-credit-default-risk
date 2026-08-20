"""`HC-M4-08` — the scoring decision. Pure business logic: no pandas, no
sklearn, no framework of any kind. Given a probability, decide whether an
applicant is flagged as high-risk.

`DEFAULT_THRESHOLD` is a business rule (`HC-M3-25`'s cost-asymmetry
decision: the largest threshold that still catches at least 70% of
actual defaulters on leakage-safe out-of-fold predictions), not an
infrastructure detail -- it belongs here, in the domain, not in
`config.py` alongside file paths and tracking URIs.
"""

from dataclasses import dataclass

# HC-M3-25: chosen to satisfy a 70% recall floor on defaulters, subject
# to that floor maximizing precision. Not 0.5 -- see the notebook for why.
DEFAULT_THRESHOLD = 0.485


@dataclass(frozen=True)
class Decision:
    probability: float
    is_high_risk: bool
    threshold: float


def decide(probability: float, threshold: float = DEFAULT_THRESHOLD) -> Decision:
    return Decision(
        probability=probability,
        is_high_risk=probability >= threshold,
        threshold=threshold,
    )
