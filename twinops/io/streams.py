"""Interfacce per dati batch o streaming (sensori real-time)."""

from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np


class BatchStream:
    """
    Stream batch: sequenza pre-caricata di (u, y) per replay o simulazione.
    """

    def __init__(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        measurements: Union[np.ndarray, List[np.ndarray]],
    ) -> None:
        """
        Args:
            inputs: array (N, input_dim) o lista di vettori u_t
            measurements: array (N, meas_dim) o lista di vettori y_t
        """
        self._u = np.atleast_2d(np.asarray(inputs))
        self._y = np.atleast_2d(np.asarray(measurements))
        if len(self._u) != len(self._y):
            raise ValueError("inputs e measurements devono avere la stessa lunghezza")
        self._index = 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self._index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._index >= len(self._u):
            raise StopIteration
        u_t = self._u[self._index]
        y_t = self._y[self._index]
        self._index += 1
        return u_t, y_t

    def __len__(self) -> int:
        return len(self._u)

    def reset(self) -> None:
        self._index = 0


class SensorStream:
    """
    Interfaccia per stream live: iterabile che restituisce (u_t, y_t).
    Implementazioni concrete possono leggere da socket, MQTT, file, ecc.
    """

    def __init__(self, source: Optional[Any] = None) -> None:
        """
        Args:
            source: sorgente dati (None = da implementare in sottoclassi).
        """
        self._source = source

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Sottoclassi o adapter devono implementare __next__")
