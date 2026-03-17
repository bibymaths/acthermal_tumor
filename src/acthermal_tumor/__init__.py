"""Top‑level package for the acthermal_tumor simulator.

This module exposes the main API surface of the package.  Users can
import the `ThermalTumorSimulator` class directly from this namespace
along with the `Parameters` dataclass and utilities for generating
initial conditions.
"""

from importlib.metadata import version as _pkg_version

from .solver import ThermalTumorSimulator
from .parameters import Parameters
from .utils import generate_initial_conditions, plot_state

__all__ = [
    "ThermalTumorSimulator",
    "Parameters",
    "generate_initial_conditions",
    "plot_state",
]

# Expose package version via __version__
try:
    __version__ = _pkg_version(__name__.split(".")[0])
except Exception:
    __version__ = "0.0.0"
