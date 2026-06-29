"""Registry of Python benchmark adapters.

Maps a stable adapter id (also the per-library JSON filename and table column) to
its class and constructor kwargs. The orchestrator iterates this in order; the
Julia/DynamicalSystems.jl column is produced by the external Julia script, not
an adapter here.
"""

from __future__ import annotations

from typing import Any

from adapters.antropy_adapter import AntropyAdapter
from adapters.dysts_adapter import DystsAdapter
from adapters.neurokit2_adapter import NeuroKit2Adapter
from adapters.nolds_adapter import NoldsAdapter
from adapters.nolitsa_adapter import NolitsaAdapter
from adapters.pynamical_adapter import PynamicalAdapter
from adapters.pyunicorn_adapter import PyunicornAdapter
from adapters.scipy_adapter import SciPyAdapter
from adapters.tsdynamics_adapter import TSDynamicsAdapter

# id -> (class, kwargs). Order is the column order in the tables.
REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "tsdynamics-interp": (TSDynamicsAdapter, {"backend": "interp"}),
    "tsdynamics-jit": (TSDynamicsAdapter, {"backend": "jit"}),
    "scipy": (SciPyAdapter, {}),
    "dysts": (DystsAdapter, {}),
    "pynamical": (PynamicalAdapter, {}),
    "nolds": (NoldsAdapter, {}),
    "nolitsa": (NolitsaAdapter, {}),
    "antropy": (AntropyAdapter, {}),
    "neurokit2": (NeuroKit2Adapter, {}),
    "pyunicorn": (PyunicornAdapter, {}),
}


def make_adapter(adapter_id: str, config: dict[str, Any]) -> Any:
    """Instantiate the adapter registered under ``adapter_id``."""
    cls, kwargs = REGISTRY[adapter_id]
    return cls(config, **kwargs)
