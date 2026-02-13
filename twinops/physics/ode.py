"""Modello a equazioni differenziali ordinarie e integratori (Euler, RK4)."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np

from twinops.core.component import TwinComponent


def euler_step(f: Callable[[np.ndarray, np.ndarray, float], np.ndarray], x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Un passo di Eulero esplicito: x_{n+1} = x_n + dt * f(x_n, u_n, t_n)."""
    return x + dt * f(x, u, t)


def rk4_step(f: Callable[[np.ndarray, np.ndarray, float], np.ndarray], x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Un passo di Runge-Kutta 4."""
    k1 = f(x, u, t)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt)
    k4 = f(x + dt * k3, u, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class EulerIntegrator:
    """Integratore Euler esplicito."""

    @staticmethod
    def step(f: Callable[[np.ndarray, np.ndarray, float], np.ndarray], x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return euler_step(f, x, u, t, dt)


class RK4Integrator:
    """Integratore Runge-Kutta 4."""

    @staticmethod
    def step(f: Callable[[np.ndarray, np.ndarray, float], np.ndarray], x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return rk4_step(f, x, u, t, dt)


class ODEModel(TwinComponent):
    """
    Classe base per modelli descritti da ODE: dx/dt = f(x, u, t).
    Il metodo rhs() va implementato nelle sottoclassi.
    """

    def __init__(self, integrator: Optional[Any] = None) -> None:
        """
        Args:
            integrator: oggetto con metodo step(f, x, u, t, dt). Default: RK4.
        """
        self.integrator = integrator or RK4Integrator()
        self._t: float = 0.0
        self._output_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Termine destro dell'ODE: dx/dt = rhs(x, u, t).
        Da implementare nelle sottoclassi.
        """
        raise NotImplementedError("Sottoclassi devono implementare rhs(x, u, t).")

    def output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Uscita osservabile (misura) in funzione di stato e ingresso.
        Default: ritorna lo stato; override per output parziale.
        """
        if self._output_fn is not None:
            return self._output_fn(x, u)
        return np.asarray(x, dtype=float)

    def set_output_fn(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """Imposta una funzione custom per l'output (misura)."""
        self._output_fn = fn

    def initialize(self, **kwargs: Any) -> None:
        state = kwargs.get("state")
        if state is not None:
            self._t = 0.0

    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        x = np.atleast_1d(state)
        u_arr = np.atleast_1d(u)
        x_next = self.integrator.step(self.rhs, x, u_arr, self._t, dt)
        self._t += dt
        y = self.output(x_next, u_arr)
        return {"state": x_next, "output": y}

    def state_dict(self) -> Dict[str, Any]:
        return {"t": self._t}
