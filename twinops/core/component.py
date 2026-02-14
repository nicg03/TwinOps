"""Base interface for digital twin components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class TwinComponent(ABC):
    """
    Base interface for all twin components:
    physics, residual model, estimator, health.
    """

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        """Initialize the component (state, parameters)."""
        pass

    @abstractmethod
    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute one time step.

        Args:
            state: current (or estimated) state
            u: control input
            dt: time step
            measurement: available measurement (optional, for estimator)
            **kwargs: additional arguments for extensions

        Returns:
            Dictionary with component output (keys depend on type).
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the component's internal state for checkpointing/serialization.
        Override for stateful components.
        """
        return {}
