"""Core: orchestratore e interfacce del digital twin."""

from twinops.core.component import TwinComponent
from twinops.core.signals import SignalSpec
from twinops.core.system import TwinSystem
from twinops.core.history import TwinHistory

__all__ = ["TwinComponent", "SignalSpec", "TwinSystem", "TwinHistory"]
