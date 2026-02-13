"""Stima Remaining Useful Life (RUL) a partire da Health Indicators."""

from typing import Any, Dict, Optional

import numpy as np

from twinops.core.component import TwinComponent


class SimpleRUL(TwinComponent):
    """
    RUL semplice: estrapolazione lineare del trend dell'HI fino a soglia di fallimento.
    RUL = (HI - HI_fail) / (dHI/dt) in step temporali, se dHI/dt < 0.
    """

    def __init__(
        self,
        hi_fail: float = 0.3,
        min_rul: float = 0.0,
        max_rul: Optional[float] = None,
    ) -> None:
        """
        Args:
            hi_fail: soglia HI sotto cui si considera guasto
            min_rul: RUL minimo restituito
            max_rul: RUL massimo (cap)
        """
        self.hi_fail = hi_fail
        self.min_rul = min_rul
        self.max_rul = max_rul
        self._hi_history: list = []
        self._dt: float = 0.01

    def initialize(self, **kwargs: Any) -> None:
        self._hi_history = []
        self._dt = kwargs.get("dt", 0.01)

    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        health_indicator: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._dt = dt
        hi = health_indicator if health_indicator is not None else 1.0
        self._hi_history.append(hi)
        # Mantieni solo ultimi N per trend
        if len(self._hi_history) > 100:
            self._hi_history = self._hi_history[-100:]
        rul = None
        if len(self._hi_history) >= 2:
            his = np.array(self._hi_history[-20:])
            t = np.arange(len(his), dtype=float) * self._dt
            slope = np.polyfit(t, his, 1)[0]
            if slope < -1e-10 and hi > self.hi_fail:
                # step fino a HI_fail: (hi - hi_fail) / (-slope) in secondi -> step
                steps_to_fail = (hi - self.hi_fail) / (-slope) if slope != 0 else 1e6
                rul = max(self.min_rul, float(steps_to_fail * self._dt))
                if self.max_rul is not None:
                    rul = min(rul, self.max_rul)
            elif hi <= self.hi_fail:
                rul = self.min_rul
        return {"rul": rul}

    def state_dict(self) -> Dict[str, Any]:
        return {"hi_history": self._hi_history.copy()}
