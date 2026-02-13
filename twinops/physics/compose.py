"""
Physics model composition: Series and Parallel.

Allows building complex twins by combining ODEModel (or TwinComponent
with step/initialize) without rewriting equations.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from twinops.core.component import TwinComponent


def _ensure_1d(a: Union[list, np.ndarray], name: str) -> np.ndarray:
    out = np.atleast_1d(np.asarray(a, dtype=float))
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {out.shape}")
    return out


class SeriesModel(TwinComponent):
    """
    Two models in series: first model's output is second model's input.

    Composite state: x = [x1, x2] with x1 of dimension state_dim1, x2 of state_dim2.
    External input u goes to first model; u2 = connection_fn(y1, u) goes to second.
    Default connection_fn: u2 = y1 (second receives first's output).
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
            model1: first model (input u, output y1).
            model2: second model (input u2, output y2).
            state_dim1: state dimension of model1.
            state_dim2: state dimension of model2.
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
                    f"state must have length {self.state_dim1 + self.state_dim2}, "
                    f"got {len(state)}"
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
                f"state must have length {self.state_dim1 + self.state_dim2}, "
                f"got {len(state)}"
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
    Two models in parallel: same input u to both, state and output concatenated.

    Composite state: x = [x1, x2]. Output: y = [y1, y2].
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
            model1: first model.
            model2: second model.
            state_dim1: state dimension of model1.
            state_dim2: state dimension of model2.
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
                    f"state must have length {self.state_dim1 + self.state_dim2}, "
                    f"got {len(state)}"
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
                f"state must have length {self.state_dim1 + self.state_dim2}, "
                f"got {len(state)}"
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
