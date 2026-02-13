"""
Composizione di modelli fisici: Serie e Parallelo.

Permette di costruire twin complessi combinando ODEModel (o TwinComponent
con step/initialize) senza riscrivere le equazioni.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from twinops.core.component import TwinComponent


def _ensure_1d(a: Union[list, np.ndarray], name: str) -> np.ndarray:
    out = np.atleast_1d(np.asarray(a, dtype=float))
    if out.ndim != 1:
        raise ValueError(f"{name} deve essere 1D, ricevuto shape {out.shape}")
    return out


class SeriesModel(TwinComponent):
    """
    Due modelli in serie: uscita del primo è ingresso del secondo.

    Stato composto: x = [x1, x2] con x1 di dimensione state_dim1, x2 di state_dim2.
    Ingresso esterno u va al primo modello; u2 = connection_fn(y1, u) va al secondo.
    Default connection_fn: u2 = y1 (il secondo riceve l'uscita del primo).
    """

    def __init__(
        self,
        model1: TwinComponent,
        model2: TwinComponent,
        state_dim1: int,
        state_dim2: int,
        connection_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        Args:
            model1: primo modello (ingresso u, uscita y1).
            model2: secondo modello (ingresso u2, uscita y2).
            state_dim1: dimensione stato di model1.
            state_dim2: dimensione stato di model2.
            connection_fn: (y1, u) -> u2. Default: u2 = y1.
        """
        self.model1 = model1
        self.model2 = model2
        self.state_dim1 = state_dim1
        self.state_dim2 = state_dim2
        self._connection_fn = connection_fn or (lambda y1, u: np.atleast_1d(y1))

    def initialize(self, **kwargs: Any) -> None:
        state = kwargs.get("state")
        kwargs_no_state = {k: v for k, v in kwargs.items() if k != "state"}
        if state is not None:
            state = _ensure_1d(state, "state")
            if len(state) != self.state_dim1 + self.state_dim2:
                raise ValueError(
                    f"state deve avere lunghezza {self.state_dim1 + self.state_dim2}, "
                    f"ricevuto {len(state)}"
                )
            x1 = state[: self.state_dim1]
            x2 = state[self.state_dim1 :]
            self.model1.initialize(state=x1, **kwargs_no_state)
            self.model2.initialize(state=x2, **kwargs_no_state)
        else:
            self.model1.initialize(**kwargs_no_state)
            self.model2.initialize(**kwargs_no_state)

    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        state = _ensure_1d(state, "state")
        u_arr = _ensure_1d(u, "u")
        if len(state) != self.state_dim1 + self.state_dim2:
            raise ValueError(
                f"state deve avere lunghezza {self.state_dim1 + self.state_dim2}, "
                f"ricevuto {len(state)}"
            )
        x1 = state[: self.state_dim1]
        x2 = state[self.state_dim1 :]

        out1 = self.model1.step(state=x1, u=u_arr, dt=dt, measurement=measurement, **kwargs)
        x1_next = np.atleast_1d(out1["state"])
        y1 = np.atleast_1d(out1.get("output", x1_next))

        u2 = self._connection_fn(y1, u_arr)
        u2 = np.atleast_1d(u2)
        out2 = self.model2.step(state=x2, u=u2, dt=dt, **kwargs)
        x2_next = np.atleast_1d(out2["state"])
        y2 = np.atleast_1d(out2.get("output", x2_next))

        state_next = np.concatenate([x1_next, x2_next])
        output = np.concatenate([y1, y2])
        return {"state": state_next, "output": output}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model1": self.model1.state_dict(),
            "model2": self.model2.state_dict(),
        }


class ParallelModel(TwinComponent):
    """
    Due modelli in parallelo: stesso ingresso u a entrambi, stato e uscita concatenati.

    Stato composto: x = [x1, x2]. Output: y = [y1, y2].
    """

    def __init__(
        self,
        model1: TwinComponent,
        model2: TwinComponent,
        state_dim1: int,
        state_dim2: int,
    ) -> None:
        """
        Args:
            model1: primo modello.
            model2: secondo modello.
            state_dim1: dimensione stato di model1.
            state_dim2: dimensione stato di model2.
        """
        self.model1 = model1
        self.model2 = model2
        self.state_dim1 = state_dim1
        self.state_dim2 = state_dim2

    def initialize(self, **kwargs: Any) -> None:
        state = kwargs.get("state")
        kwargs_no_state = {k: v for k, v in kwargs.items() if k != "state"}
        if state is not None:
            state = _ensure_1d(state, "state")
            if len(state) != self.state_dim1 + self.state_dim2:
                raise ValueError(
                    f"state deve avere lunghezza {self.state_dim1 + self.state_dim2}, "
                    f"ricevuto {len(state)}"
                )
            x1 = state[: self.state_dim1]
            x2 = state[self.state_dim1 :]
            self.model1.initialize(state=x1, **kwargs_no_state)
            self.model2.initialize(state=x2, **kwargs_no_state)
        else:
            self.model1.initialize(**kwargs_no_state)
            self.model2.initialize(**kwargs_no_state)

    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        state = _ensure_1d(state, "state")
        u_arr = _ensure_1d(u, "u")
        if len(state) != self.state_dim1 + self.state_dim2:
            raise ValueError(
                f"state deve avere lunghezza {self.state_dim1 + self.state_dim2}, "
                f"ricevuto {len(state)}"
            )
        x1 = state[: self.state_dim1]
        x2 = state[self.state_dim1 :]

        out1 = self.model1.step(state=x1, u=u_arr, dt=dt, measurement=measurement, **kwargs)
        out2 = self.model2.step(state=x2, u=u_arr, dt=dt, **kwargs)

        x1_next = np.atleast_1d(out1["state"])
        x2_next = np.atleast_1d(out2["state"])
        y1 = np.atleast_1d(out1.get("output", x1_next))
        y2 = np.atleast_1d(out2.get("output", x2_next))

        state_next = np.concatenate([x1_next, x2_next])
        output = np.concatenate([y1, y2])
        return {"state": state_next, "output": output}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model1": self.model1.state_dict(),
            "model2": self.model2.state_dict(),
        }
