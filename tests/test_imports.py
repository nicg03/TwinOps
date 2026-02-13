"""Verify that main modules are importable."""

import pytest


def test_import_twinops() -> None:
    import twinops
    assert twinops.__version__ == "0.1.0"


def test_import_core() -> None:
    from twinops.core import TwinSystem, TwinComponent, SignalSpec, TwinHistory
    assert TwinSystem is not None
    assert TwinComponent is not None
    assert SignalSpec is not None
    assert TwinHistory is not None


def test_import_physics() -> None:
    from twinops.physics import ODEModel, SymbolicODEModel, EulerIntegrator, RK4Integrator
    assert ODEModel is not None
    assert SymbolicODEModel is not None
    assert EulerIntegrator is not None
    assert RK4Integrator is not None


def test_import_estimation() -> None:
    from twinops.estimation import EKF, AnomalyDetector
    assert EKF is not None
    assert AnomalyDetector is not None


def test_import_health() -> None:
    from twinops.health import HealthIndicator, SimpleRUL
    assert HealthIndicator is not None
    assert SimpleRUL is not None


def test_import_io() -> None:
    from twinops.io import BatchStream, SensorStream, save_config, load_config
    assert BatchStream is not None
    assert SensorStream is not None
    assert save_config is not None
    assert load_config is not None


def test_signal_spec_size() -> None:
    from twinops.core.signals import SignalSpec
    s = SignalSpec("x", (2, 3), unit="m")
    assert s.size() == 6
