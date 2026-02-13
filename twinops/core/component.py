"""Interfaccia base per i componenti del digital twin."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class TwinComponent(ABC):
    """
    Interfaccia base per tutti i componenti del twin:
    fisica, modello residuale, estimatore, health/RUL.
    """

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        """Inizializza il componente (stato, parametri)."""
        pass

    @abstractmethod
    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Esegue un passo temporale.

        Args:
            state: stato corrente (o stimato)
            u: ingresso di controllo
            dt: passo temporale
            measurement: misura disponibile (opzionale, per estimator)
            **kwargs: argomenti aggiuntivi per estensioni

        Returns:
            Dizionario con output del componente (chiavi dipendono dal tipo).
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """
        Restituisce lo stato interno del componente per checkpoint/serializzazione.
        Override per componenti stateful.
        """
        return {}
