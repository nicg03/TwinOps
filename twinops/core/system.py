"""Digital twin orchestrator: time loop and component integration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from twinops.core.component import TwinComponent
from twinops.core.history import TwinHistory


@dataclass
class StepResult:
    """Result of one twin step: estimated state, anomaly, RUL, etc."""

    state: np.ndarray
    covariance: Optional[np.ndarray] = None
    anomaly: float = 0.0
    rul: Optional[float] = None
    health_indicator: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


class TwinSystem:
    """
    Digital twin orchestrator.
    Handles the time loop: physics -> ML residual -> estimator -> health/RUL.
    """

    def __init__(
        self,
        physics: TwinComponent,
        estimator: TwinComponent,
        residual: Optional[TwinComponent] = None,
        health: Optional[TwinComponent] = None,
        rul: Optional[TwinComponent] = None,
        dt: float = 0.01,
        state_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        meas_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            physics: physics model (ODE or FMU)
            estimator: filter for state/parameter estimation (e.g. EKF)
            residual: ML corrector (optional)
            health: health indicator computation (optional)
            rul: RUL estimation (optional)
            dt: time step
            state_dim: state dimension (default from physics)
            input_dim: input dimension
            meas_dim: measurement dimension
        """
        self.physics = physics
        self.estimator = estimator
        self.residual = residual
        self.health = health
        self.rul = rul
        self.dt = dt
        self._state_dim = state_dim
        self._input_dim = input_dim
        self._meas_dim = meas_dim
        self._x: Optional[np.ndarray] = None
        self._initialized = False
        self._time: float = 0.0

    def initialize(
        self,
        x0: Union[List[float], np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Initialize the twin with initial state x0."""
        self._x = np.atleast_1d(np.asarray(x0, dtype=float))
        self._time = 0.0
        self.physics.initialize(state=self._x.copy(), **kwargs)
        self.estimator.initialize(state=self._x.copy(), **kwargs)
        if self.residual is not None:
            self.residual.initialize(**kwargs)
        if self.health is not None:
            self.health.initialize(**kwargs)
        if self.rul is not None:
            self.rul.initialize(**kwargs)
        self._initialized = True

    def step(
        self,
        u: Union[List[float], np.ndarray],
        measurement: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs: Any,
    ) -> StepResult:
        """
        Execute one step: physics prediction (+ residual), update with measurement, anomaly/RUL.

        Args:
            u: control input
            measurement: sensor measurement (optional)
            **kwargs: additional arguments for components

        Returns:
            StepResult with estimated state, anomaly score, RUL, etc.
        """
        if not self._initialized or self._x is None:
            raise RuntimeError("Twin not initialized: call initialize(x0) before step().")
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        meas = np.atleast_1d(np.asarray(measurement, dtype=float)) if measurement is not None else None

        # 1) Fisica
        physics_out = self.physics.step(state=self._x, u=u_arr, dt=self.dt, **kwargs)
        x_pred = physics_out.get("state", self._x.copy())
        y_physics = physics_out.get("output", np.array([]))

        # 2) ML residual (correction)
        if self.residual is not None:
            res_out = self.residual.step(state=x_pred, u=u_arr, dt=self.dt, **kwargs)
            correction = res_out.get("correction", np.zeros_like(x_pred))
            x_pred = x_pred + np.atleast_1d(correction)

        # 3) Estimator (data assimilation)
        est_out = self.estimator.step(
            state=x_pred,
            u=u_arr,
            dt=self.dt,
            measurement=meas,
            physics_output=physics_out,
            **kwargs,
        )
        self._x = est_out.get("state", x_pred)
        covariance = est_out.get("covariance")
        anomaly = float(est_out.get("anomaly", 0.0))

        # 4) Health indicator
        hi = None
        if self.health is not None:
            health_out = self.health.step(
                state=self._x,
                u=u_arr,
                dt=self.dt,
                measurement=meas,
                anomaly=anomaly,
                **kwargs,
            )
            hi = health_out.get("health_indicator")

        # 5) RUL
        rul_val = None
        if self.rul is not None:
            rul_out = self.rul.step(
                state=self._x,
                u=u_arr,
                dt=self.dt,
                health_indicator=hi,
                **kwargs,
            )
            rul_val = rul_out.get("rul")

        self._time += self.dt
        return StepResult(
            state=self._x.copy(),
            covariance=covariance,
            anomaly=anomaly,
            rul=rul_val,
            health_indicator=hi,
            extra={"physics_output": y_physics, "estimator_out": est_out},
        )

    @property
    def state(self) -> Optional[np.ndarray]:
        """Current estimated state."""
        return self._x.copy() if self._x is not None else None

    @property
    def time(self) -> float:
        """Current simulated time."""
        return self._time

    def state_dict(self) -> Dict[str, Any]:
        """Full twin state for checkpointing."""
        return {
            "state": self._x.copy() if self._x is not None else None,
            "time": self._time,
            "physics": self.physics.state_dict(),
            "estimator": self.estimator.state_dict(),
            "residual": self.residual.state_dict() if self.residual else {},
            "health": self.health.state_dict() if self.health else {},
            "rul": self.rul.state_dict() if self.rul else {},
        }
