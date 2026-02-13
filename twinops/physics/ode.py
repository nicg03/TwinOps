"""
Modello base a equazioni differenziali ordinarie: dx/dt = rhs(x, u, t).

Usa un integratore (da physics.integrators) per avanzare nel tempo.
Il metodo rhs() va implementato nelle sottoclassi.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from twinops.core.component import TwinComponent

from twinops.physics.integrators import RK4Integrator


class ODEModel(TwinComponent):
    """
    Classe base per modelli ODE: dx/dt = rhs(x, u, t).
    Sottoclassi implementano rhs(); l'integratore è configurabile.
    """

    def __init__(self, integrator: Optional[Any] = None) -> None:
        """
        Args:
            integrator: oggetto con metodo step(f, x, u, t, dt). Default: RK4Integrator.
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
        Default: ritorna lo stato; override o set_output_fn per output parziale.
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    class HarmonicOscillator(ODEModel):
        def __init__(self, omega: float = 1.0, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.omega_sq = omega ** 2

        def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
            x1, x2 = x[0], x[1]
            return np.array([x2, -self.omega_sq * x1])

    omega = 1.0
    dt = 0.05
    t_end = 10.0
    x0 = np.array([1.0, 0.0])

    model = HarmonicOscillator(omega=omega)
    model.initialize()

    t_vals = [0.0]
    x_vals = [x0.copy()]
    x = x0.copy()
    t = 0.0
    while t < t_end:
        out = model.step(state=x, u=np.array([]), dt=dt)
        x = out["state"]
        t += dt
        t_vals.append(t)
        x_vals.append(x.copy())

    t_vals = np.array(t_vals)
    pos = np.array([xv[0] for xv in x_vals])
    vel = np.array([xv[1] for xv in x_vals])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t_vals, pos, label="posizione x1")
    ax1.set_ylabel("x1")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(t_vals, vel, label="velocità x2")
    ax2.set_ylabel("x2")
    ax2.set_xlabel("t")
    ax2.legend()
    ax2.grid(True)
    plt.suptitle("Oscillatore armonico (ODE)")
    plt.tight_layout()
    plt.show()
