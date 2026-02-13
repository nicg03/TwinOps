"""
Modello fisico ODE appreso da dati tramite symbolic regression (genetic programming).

SymbolicODEModel è un ODEModel il cui rhs(x, u, t) è dato da espressioni simboliche
apprese da serie temporali (es. CSV). Funge solo da modello fisico: stesso contratto
di ODEModel (initialize, step, rhs), utilizzabile nel twin al posto di modelli
parametrici (FirstOrderLag, PumpLike, ecc.).

Richiede dipendenza opzionale: pip install twinops[symbolic]  # installa gplearn
"""

from typing import Any, Dict, List, Optional

import numpy as np

from twinops.physics.ode import ODEModel

try:
    from gplearn.genetic import SymbolicRegressor
except ImportError:
    SymbolicRegressor = None  # type: ignore[misc, assignment]


def _compute_dx_dt_central(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Derivate temporali con differenze centrali (interno), forward/backward agli estremi.
    t: (n_steps,), x: (n_steps, n_states) -> (n_steps, n_states)
    """
    n_steps, n_states = x.shape
    dx_dt = np.empty_like(x)
    dt = np.diff(t)
    if dt.size == 0:
        return dx_dt
    for j in range(n_states):
        dx_dt[0, j] = (x[1, j] - x[0, j]) / dt[0] if n_steps > 1 else 0.0
        for i in range(1, n_steps - 1):
            dx_dt[i, j] = (x[i + 1, j] - x[i - 1, j]) / (t[i + 1] - t[i - 1])
        dx_dt[n_steps - 1, j] = (
            (x[n_steps - 1, j] - x[n_steps - 2, j]) / dt[-1] if n_steps > 1 else 0.0
        )
    return dx_dt


def _build_design_matrix(
    x: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    n_states: int,
    n_inputs: int,
) -> np.ndarray:
    """Matrice delle features: [x_1..x_n, u_1..u_m, t]. Shape (n_samples, n_states + n_inputs + 1)."""
    n_steps = x.shape[0]
    cols: List[np.ndarray] = []
    for j in range(n_states):
        cols.append(x[:, j].reshape(-1, 1))
    for j in range(n_inputs):
        u_j = u[:, j] if u.ndim > 1 else u[:]
        cols.append(u_j.reshape(-1, 1))
    cols.append(np.asarray(t, dtype=float).reshape(-1, 1))
    return np.hstack(cols)


class SymbolicODEModel(ODEModel):
    """
    Modello ODE il cui rhs è appreso da dati tramite symbolic regression (gplearn).

    Dopo fit_from_timeseries(t, x, u) o fit(X, y), rhs(x, u, t) valuta le
    espressioni apprese per ogni componente di stato. Si usa come qualsiasi
    altro ODEModel nel twin (solo come modello fisico).
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int = 0,
        integrator: Optional[Any] = None,
        *,
        population_size: int = 500,
        generations: int = 20,
        function_set: tuple = ("add", "sub", "mul", "div"),
        random_state: Optional[int] = None,
        **symreg_kwargs: Any,
    ) -> None:
        """
        Args:
            n_states: numero di variabili di stato.
            n_inputs: numero di ingressi (default 0).
            integrator: integratore ODE (default RK4).
            population_size: popolazione per il genetic programming.
            generations: generazioni di evoluzione.
            function_set: operazioni simboliche (add, sub, mul, div, sqrt, ...).
            random_state: seed per riproducibilità.
            **symreg_kwargs: altri argomenti per gplearn.genetic.SymbolicRegressor.
        """
        super().__init__(integrator=integrator)
        if SymbolicRegressor is None:
            raise ImportError(
                "SymbolicODEModel richiede gplearn. Installa con: pip install twinops[symbolic]"
            )
        self.n_states = n_states
        self.n_inputs = n_inputs
        self._regressors: List[Any] = []  # uno per ogni stato
        self._feature_names: List[str] = []
        self._symreg_kwargs = {
            "population_size": population_size,
            "generations": generations,
            "function_set": function_set,
            "random_state": random_state,
            **symreg_kwargs,
        }
        self._build_feature_names()

    def _build_feature_names(self) -> None:
        names: List[str] = []
        for i in range(self.n_states):
            names.append(f"x{i}")
        for i in range(self.n_inputs):
            names.append(f"u{i}")
        names.append("t")
        self._feature_names = names

    def fit_from_timeseries(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: Optional[np.ndarray] = None,
    ) -> "SymbolicODEModel":
        """
        Apprende rhs da serie temporali (t, x, u).
        Le derivate dx/dt sono stimate con differenze centrali.

        Args:
            t: tempi, shape (n_steps,).
            x: stati, shape (n_steps, n_states).
            u: ingressi, shape (n_steps,) o (n_steps, n_inputs). Se None, usa zeri.

        Returns:
            self (per method chaining).
        """
        t = np.asarray(t, dtype=float).ravel()
        x = np.asarray(x, dtype=float)
        n_steps = x.shape[0]
        if x.shape[1] != self.n_states:
            raise ValueError(
                f"x deve avere n_states={self.n_states} colonne, ricevuto {x.shape[1]}"
            )
        if u is None:
            u = np.zeros((n_steps, self.n_inputs))
        else:
            u = np.asarray(u, dtype=float)
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            if u.shape[0] != n_steps or u.shape[1] < self.n_inputs:
                raise ValueError(
                    f"u deve avere shape (n_steps, n_inputs)=({n_steps}, {self.n_inputs})"
                )
        dx_dt = _compute_dx_dt_central(t, x)
        X = _build_design_matrix(x, u, t, self.n_states, self.n_inputs)
        return self.fit(X, dx_dt)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicODEModel":
        """
        Apprende rhs da matrice di design e derivate.

        Args:
            X: features, shape (n_samples, n_states + n_inputs + 1),
               colonne [x_0..x_{n_states-1}, u_0.., t].
            y: derivate dx/dt, shape (n_samples, n_states).

        Returns:
            self.
        """
        n_features = self.n_states + self.n_inputs + 1
        if X.shape[1] != n_features:
            raise ValueError(
                f"X deve avere {n_features} colonne (stati + ingressi + t), ricevuto {X.shape[1]}"
            )
        if y.shape[1] != self.n_states:
            raise ValueError(
                f"y deve avere n_states={self.n_states} colonne, ricevuto {y.shape[1]}"
            )
        self._regressors = []
        for i in range(self.n_states):
            reg = SymbolicRegressor(
                feature_names=self._feature_names,
                **self._symreg_kwargs,
            )
            reg.fit(X, y[:, i])
            self._regressors.append(reg)
        return self

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """Termine destro ODE: dx/dt = rhs(x, u, t). Valuta le espressioni apprese."""
        if not self._regressors:
            raise RuntimeError(
                "SymbolicODEModel non ancora addestrato. Chiamare fit() o fit_from_timeseries()."
            )
        x = np.atleast_1d(x)
        u = np.atleast_1d(u)
        if u.size < self.n_inputs:
            u = np.resize(u, self.n_inputs)
        row = _build_design_matrix(
            x.reshape(1, -1),
            u.reshape(1, -1),
            np.array([t]),
            self.n_states,
            self.n_inputs,
        )
        out = np.array(
            [self._regressors[i].predict(row)[0] for i in range(self.n_states)],
            dtype=float,
        )
        return out

    def get_expressions(self) -> List[str]:
        """
        Restituisce le espressioni simboliche apprese per ogni componente di dx/dt.
        Utile per interpretabilità (modello fisico "leggibile").
        """
        if not self._regressors:
            return []
        return [str(reg._program) for reg in self._regressors]

    def state_dict(self) -> Dict[str, Any]:
        d = super().state_dict()
        d["n_states"] = self.n_states
        d["n_inputs"] = self.n_inputs
        d["expressions"] = self.get_expressions()
        return d
