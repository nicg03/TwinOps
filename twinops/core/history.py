"""Logging in memoria e export (CSV, numpy) per il digital twin."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class TwinHistory:
    """
    Buffer in memoria per stati, ingressi, uscite e metriche del twin.
    Supporta export in CSV e numpy.
    """

    def __init__(self, max_length: Optional[int] = None) -> None:
        """
        Args:
            max_length: numero massimo di step da tenere (None = illimitato).
        """
        self._max_length = max_length
        self._data: Dict[str, List[Any]] = {}
        self._step_count = 0

    def append(self, **kwargs: Any) -> None:
        """Aggiunge un record per lo step corrente (chiave -> valore)."""
        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(value)
        self._step_count += 1
        if self._max_length is not None and self._step_count > self._max_length:
            for key in self._data:
                self._data[key] = self._data[key][-self._max_length:]
            self._step_count = self._max_length

    def clear(self) -> None:
        """Svuota la history."""
        self._data.clear()
        self._step_count = 0

    def get(self, key: str) -> np.ndarray:
        """Restituisce la serie per una chiave come array numpy."""
        if key not in self._data:
            return np.array([])
        return np.array(self._data[key])

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Restituisce tutti i dati come dizionario di array."""
        return {k: self.get(k) for k in self._data}

    def to_numpy(self, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Come to_dict ma può filtrare per chiavi."""
        keys = keys or list(self._data.keys())
        return {k: self.get(k) for k in keys if k in self._data}

    def to_csv(
        self,
        path: Union[str, Path],
        keys: Optional[List[str]] = None,
        delimiter: str = ",",
    ) -> None:
        """
        Esporta in CSV. Le chiavi diventano colonne; ogni riga è uno step.
        """
        path = Path(path)
        keys = keys or list(self._data.keys())
        if not keys:
            path.write_text("")
            return
        arrays = [self.get(k) for k in keys]
        # Assicura stessa lunghezza
        n = max(len(a) for a in arrays)
        rows = []
        for i in range(n):
            row = []
            for a in arrays:
                if i < len(a):
                    v = a[i]
                    row.append(str(v) if not isinstance(v, (list, np.ndarray)) else str(v).replace(" ", ","))
                else:
                    row.append("")
            rows.append(delimiter.join(row))
        header = delimiter.join(keys)
        path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")

    def __len__(self) -> int:
        return self._step_count
