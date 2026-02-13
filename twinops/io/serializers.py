"""Salvataggio e caricamento di configurazioni e snapshot del twin."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def _numpy_encoder(obj: Any) -> Any:
    """Converte array numpy in liste per JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Salva una configurazione (dict) in JSON.
    Array numpy vengono convertiti in liste.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Converti numpy per serializzazione
    def convert(d: Any) -> Any:
        if isinstance(d, np.ndarray):
            return d.tolist()
        if isinstance(d, dict):
            return {k: convert(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [convert(x) for x in d]
        if isinstance(d, (np.floating, np.integer)):
            return float(d) if isinstance(d, np.floating) else int(d)
        return d

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(config), f, indent=2, ensure_ascii=False)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Carica una configurazione da JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(state_dict: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Salva uno snapshot del twin (state_dict) in .npz per array e JSON per metadati.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_arrays = {}
    meta = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            np_arrays[k] = v
        else:
            meta[k] = v
    np.savez(path.with_suffix(".npz"), **np_arrays)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    save_config(meta, meta_path)


def load_snapshot(path: Union[str, Path]) -> Dict[str, Any]:
    """Carica snapshot da .npz + .meta.json."""
    path = Path(path)
    data = dict(np.load(path.with_suffix(".npz"), allow_pickle=True))
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        data.update(load_config(meta_path))
    return data
