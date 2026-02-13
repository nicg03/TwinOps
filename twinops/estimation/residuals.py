"""Calcolo anomaly score, trend, soglie (EMA, CUSUM, ecc.)."""

from typing import Optional

import numpy as np


class AnomalyDetector:
    """
    Calcola anomaly score e trend da residui di misura o da segnali scalari.
    Supporta EMA (Exponential Moving Average) e CUSUM per soglie adattive.
    """

    def __init__(
        self,
        alpha_ema: float = 0.1,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.0,
    ) -> None:
        """
        Args:
            alpha_ema: fattore smoothing EMA (0 < alpha <= 1)
            cusum_threshold: soglia CUSUM per allarme
            cusum_drift: drift per CUSUM (tipicamente 0)
        """
        self.alpha_ema = alpha_ema
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self._ema: Optional[float] = None
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0

    def update_ema(self, value: float) -> float:
        """Aggiorna EMA e restituisce il valore smoothed."""
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self.alpha_ema * value + (1 - self.alpha_ema) * self._ema
        return self._ema

    def update_cusum(self, value: float) -> tuple:
        """
        Aggiorna CUSUM. Restituisce (cusum_positive, cusum_negative).
        Allarme se uno supera cusum_threshold.
        """
        k = self.cusum_drift
        self._cusum_pos = max(0.0, self._cusum_pos + value - k)
        self._cusum_neg = max(0.0, self._cusum_neg - value - k)
        return self._cusum_pos, self._cusum_neg

    def anomaly_score(self, residual: np.ndarray) -> float:
        """Score di anomalia da vettore residuo (es. norma L2)."""
        return float(np.sqrt(np.sum(np.asarray(residual) ** 2)))

    def reset(self) -> None:
        """Reset stato interno (EMA, CUSUM)."""
        self._ema = None
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
