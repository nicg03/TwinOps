"""Definizione standardizzata di ingressi, uscite e stati (nomi, shape, unità)."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SignalSpec:
    """Specifica di un segnale: nome, shape, unità di misura."""

    name: str
    shape: Tuple[int, ...]
    unit: str = ""
    description: str = ""

    def size(self) -> int:
        """Numero totale di elementi (prodotto delle dimensioni)."""
        p = 1
        for d in self.shape:
            p *= d
        return p

    def __post_init__(self) -> None:
        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)
