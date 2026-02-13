"""Extended Kalman Filter per stima stato e parametri degradanti."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from twinops.core.component import TwinComponent


class EKF(TwinComponent):
    """
    EKF discreto: prediction con modello (fisica + eventuale residuo),
    update con misura. Supporta stima stato e parametri (stato esteso).
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        f: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None,
        h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            state_dim: dimensione stato (eventualmente esteso con parametri)
            meas_dim: dimensione misura
            f: modello di transizione x_{k+1} = f(x_k, u_k, dt). Se None, identity.
            h: modello di misura y = h(x). Se None, prime meas_dim componenti di x.
            Q: covarianza rumore processo (state_dim x state_dim)
            R: covarianza rumore misura (meas_dim x meas_dim)
            P0: covarianza iniziale (state_dim x state_dim)
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self._f = f
        self._h = h or (lambda x: np.asarray(x).flat[:meas_dim])
        self.Q = Q if Q is not None else np.eye(state_dim) * 1e-4
        self.R = R if R is not None else np.eye(meas_dim) * 1e-2
        self.P0 = P0 if P0 is not None else np.eye(state_dim) * 0.1
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._last_anomaly: float = 0.0

    def _predict(self, x: np.ndarray, u: np.ndarray, dt: float) -> tuple:
        if self._f is not None:
            x_pred = np.atleast_1d(self._f(x, u, dt))
        else:
            x_pred = np.asarray(x, dtype=float).copy()
        # Linearizzazione F = I se f non fornita; altrimenti approssimazione numerica
        F = self._jacobian_f(x, u, dt) if self._f is not None else np.eye(self.state_dim)
        P_pred = F @ self._P @ F.T + self.Q
        return x_pred, P_pred

    def _jacobian_f(self, x: np.ndarray, u: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
        """Jacobiano di f rispetto a x (approssimazione numerica)."""
        x = np.atleast_1d(x)
        f0 = self._f(x, u, dt)
        F = np.zeros((self.state_dim, self.state_dim))
        for j in range(self.state_dim):
            x_plus = x.copy()
            x_plus[j] += eps
            F[:, j] = (self._f(x_plus, u, dt) - f0) / eps
        return F

    def _jacobian_h(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Jacobiano di h rispetto a x."""
        x = np.atleast_1d(x)
        h0 = np.atleast_1d(self._h(x))
        H = np.zeros((self.meas_dim, self.state_dim))
        for j in range(min(self.state_dim, x.size)):
            x_plus = x.copy()
            x_plus[j] += eps
            H[:, j] = (np.atleast_1d(self._h(x_plus)) - h0) / eps
        return H

    def initialize(self, **kwargs: Any) -> None:
        state = kwargs.get("state")
        if state is not None:
            self._x = np.atleast_1d(state).astype(float)
            if self._x.size != self.state_dim:
                self._x = np.resize(self._x, self.state_dim)
        else:
            self._x = np.zeros(self.state_dim)
        self._P = np.asarray(self.P0, dtype=float).copy()
        if self._P.shape != (self.state_dim, self.state_dim):
            self._P = np.eye(self.state_dim) * (np.trace(self._P) / max(1, self._P.shape[0]))
        self._last_anomaly = 0.0

    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        physics_output: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._x = np.atleast_1d(state).astype(float)
        if self._x.size != self.state_dim:
            self._x = np.resize(self._x, self.state_dim)
        u_arr = np.atleast_1d(u)
        x_pred, P_pred = self._predict(self._x, u_arr, dt)
        self._x = x_pred
        self._P = P_pred
        anomaly = 0.0
        if measurement is not None and measurement.size >= self.meas_dim:
            y = np.atleast_1d(measurement).astype(float)[: self.meas_dim]
            H = self._jacobian_h(self._x)
            y_hat = np.atleast_1d(self._h(self._x))[: self.meas_dim]
            innov = y - y_hat
            S = H @ self._P @ H.T + self.R
            S = 0.5 * (S + S.T) + np.eye(self.meas_dim) * 1e-10
            try:
                K = self._P @ H.T @ np.linalg.solve(S, np.eye(self.meas_dim))
            except np.linalg.LinAlgError:
                K = np.zeros((self.state_dim, self.meas_dim))
            self._x = self._x + K @ innov
            self._P = (np.eye(self.state_dim) - K @ H) @ self._P
            anomaly = float(np.sqrt(np.sum(innov ** 2)))
            self._last_anomaly = anomaly
        return {
            "state": self._x.copy(),
            "covariance": self._P.copy(),
            "anomaly": anomaly if measurement is not None else self._last_anomaly,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {"x": self._x.copy() if self._x is not None else None, "P": self._P.copy() if self._P is not None else None}
