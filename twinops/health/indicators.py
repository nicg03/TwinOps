"""Trasformazione di parametri stimati o residui in Health Indicators (HI)."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from twinops.core.component import TwinComponent


class HealthIndicator(TwinComponent):
    """
    Trasforma stato stimato, parametri o anomaly in un Health Indicator scalare.
    HI in [0, 1] o in scala libera: 1 = salute piena, 0 = guasto.
    """

    def __init__(
        self,
        fn: Optional[Callable[[np.ndarray, float], float]] = None,
        state_index: Optional[int] = None,
    ) -> None:
        """
        Args:
            fn: (state, anomaly) -> HI. Se None, usa stato o anomaly di default.
            state_index: se fn è None, HI = 1 - clip(state[state_index]) o da anomaly.
        """
        self._fn = fn
        self._state_index = state_index if state_index is not None else 0
        self._last_anomaly: float = 0.0

    def _default_hi(self, state: np.ndarray, anomaly: float) -> float:
        """HI semplice: 1 / (1 + anomaly) o da componente di stato (es. parametro degradante)."""
        if state.size > self._state_index:
            # Parametro di degradazione nello stato: più è alto, più HI scende
            deg = float(np.clip(state[self._state_index], 0, 2))
            hi = max(0.0, 1.0 - deg / 2.0)
        else:
            hi = 1.0
        # Penalizza con anomaly
        hi = hi / (1.0 + 0.1 * anomaly)
        return float(np.clip(hi, 0.0, 1.0))

    def initialize(self, **kwargs: Any) -> None:
        self._last_anomaly = 0.0

    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        anomaly: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._last_anomaly = anomaly
        state = np.atleast_1d(state)
        if self._fn is not None:
            hi = self._fn(state, anomaly)
        else:
            hi = self._default_hi(state, anomaly)
        return {"health_indicator": hi}

    def state_dict(self) -> Dict[str, Any]:
        return {"last_anomaly": self._last_anomaly}
