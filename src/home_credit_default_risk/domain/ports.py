"""Ports (`HC-M4-01`) — abstract interfaces the application layer depends on.

Clean Architecture's Dependency Rule: business logic depends on
abstractions, never on a concrete infrastructure library. `ModelRegistry`
is the seam between "get the current production model" (what the scoring
use case needs) and "how that model is actually stored and versioned"
(MLflow today; anything else tomorrow, without touching a single caller).

No third-party imports here, on purpose — that is the whole point of a
port.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelRegistry(ABC):
    """Read/write access to the single production model.

    Two distinct callers use this port for two distinct reasons, which is
    why it has two methods rather than being split further: a training
    script calls `register_and_promote` once per retrain, and the scoring
    use case calls `get_production_model` on every request. Splitting
    those into separate ports would add a second interface with no
    second implementation behind it — not justified at this size.
    """

    @abstractmethod
    def get_production_model(self) -> Any:
        """Load and return whichever model is currently aliased as production."""

    @abstractmethod
    def register_and_promote(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any],
    ) -> str:
        """Log a training run, register `model` as a new version, and
        promote that version to production.

        Returns the new version identifier.
        """
