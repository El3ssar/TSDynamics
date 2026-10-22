"""Run one importable model's catalogue diagnostics with verified identity.

This dispatch adapter leaves the numerical protocol in ``diagnostics.py``.
Requests arrive as JSON on stdin; importing an identically named built-in is
never a substitute for importing the requested class.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path


def resolve_entry(request):
    """Import exactly the requested class and check its numerical fingerprint."""
    import diagnostics

    from tsdynamics import registry

    try:
        module = importlib.import_module(request["module"])
    except ModuleNotFoundError as exc:
        if exc.name != request["module"]:
            raise
        # A source-backed model loaded by an extension host may not share that
        # host's package search paths. Recreate that exact file, never a namesake.
        spec = importlib.util.spec_from_file_location(request["module"], request["source_file"])
        if spec is None or spec.loader is None:
            raise ValueError("The requested model source cannot be imported.") from exc
        module = importlib.util.module_from_spec(spec)
        sys.modules[request["module"]] = module
        spec.loader.exec_module(module)
    cls = module
    for part in request["qualname"].split("."):
        cls = getattr(cls, part)
    actual_file = inspect.getsourcefile(cls)
    if actual_file is None or Path(actual_file).resolve() != Path(request["source_file"]):
        raise ValueError("The worker imported a different model source file.")
    entry = next((item for item in registry.all_systems(builtin=None) if item.cls is cls), None)
    if entry is None:
        raise ValueError("The requested class did not register as a concrete system.")
    if (
        entry.family != request["family"]
        or diagnostics.fingerprint(entry) != request["fingerprint"]
    ):
        raise ValueError("The worker's model or numerical implementation differs from the request.")
    return entry


def main():
    """Compute one requested model and flush a completed root search checkpoint."""
    import diagnostics

    request = json.load(sys.stdin)
    entry = resolve_entry(request)
    if entry.family == "dde" and entry.name not in diagnostics.DDE_HISTORIES:
        # A fresh DDE still receives a nonconstant, reproducible history. The
        # unchanged estimator records the full formula in its preparation data.
        diagnostics.DDE_HISTORIES[entry.name] = (0.5, 0.1, 0.2, "sin")
    record = diagnostics._worker(entry, checkpoint=True)
    print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
