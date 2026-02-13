"""
TwinOps: digital twin ibridi (fisica + ML + data assimilation).
"""

__version__ = "0.1.0"

from twinops.core.system import TwinSystem
from twinops.core.component import TwinComponent
from twinops.core.signals import SignalSpec

__all__ = [
    "__version__",
    "TwinSystem",
    "TwinComponent",
    "SignalSpec",
]
