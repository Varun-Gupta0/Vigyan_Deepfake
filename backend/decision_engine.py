from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Decision:
    label: str       # "REAL" | "FAKE" | "UNCERTAIN"
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence * 100, 1)}


def decision_engine(fake_prob: float) -> Decision:
    """
    Map a fake_probability in [0, 1] to a labelled Decision.

    Thresholds:
      fake_prob < 0.30 → REAL
      fake_prob > 0.70 → FAKE
      otherwise        → UNCERTAIN
    """
    fake_prob = max(0.0, min(1.0, float(fake_prob)))

    if fake_prob > 0.70:
        return Decision("FAKE", fake_prob)
    elif fake_prob < 0.30:
        return Decision("REAL", 1.0 - fake_prob)
    else:
        return Decision("UNCERTAIN", max(fake_prob, 1.0 - fake_prob))
