"""Catalogue diagnostics retain the evidence needed to interpret each number."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry

TOOLING = Path(__file__).resolve().parents[1] / "docs" / "_tooling"


@pytest.fixture
def diagnostics(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLING))
    return importlib.import_module("diagnostics")


def test_recorded_dataset_covers_the_live_catalogue(diagnostics) -> None:
    records = diagnostics.load_records()
    assert set(records) == {entry.name for entry in registry.all_systems()}
    for name, record in records.items():
        assert record["name"] == name
        for key in ("lyapunov_spectrum", "kaplan_yorke", "equilibria"):
            card = record[key]
            assert card["status"] in {"measured", "reference", "not_applicable", "unavailable"}
            if card["status"] == "measured":
                assert np.isfinite(card["value"]).all()
                assert "settings" in card
            else:
                assert "value" not in card
                assert card["reason"]


def test_partial_leading_spectrum_does_not_turn_its_length_into_a_dimension(diagnostics) -> None:
    result = diagnostics.kaplan_yorke({"status": "measured", "value": [0.2], "complete": False})
    assert result["status"] == "unavailable"
    assert "value" not in result


def test_enough_leading_directions_can_close_a_kaplan_yorke_estimate(diagnostics) -> None:
    result = diagnostics.kaplan_yorke(
        {"status": "measured", "value": [0.2, 0.0, -1.0], "complete": False}
    )
    assert result["value"] == pytest.approx(2.2)


def test_forcing_phase_must_be_included_before_reporting_an_attractor_dimension(
    diagnostics,
) -> None:
    result = diagnostics.kaplan_yorke(
        {"status": "measured", "value": [0.2, -1.0], "complete": True}, forced=True
    )
    assert result["status"] == "not_applicable"
    assert "value" not in result


def test_reference_conditions_are_retained_and_never_relabelled_as_default_values(
    diagnostics,
) -> None:
    entry = SimpleNamespace(
        cls=type(
            "ReferenceProbe",
            (),
            {
                "_known_lyapunov": {
                    "spectrum": [0.5],
                    "params": {"a": 3.0},
                    "ic": [0.2],
                    "source": "Independent reference",
                }
            },
        ),
        reference="A published parameter-specific result",
    )
    result = diagnostics.reference(entry)
    assert result["status"] == "reference"
    assert result["params"] == {"a": 3.0}
    assert result["ic"] == [0.2]
    assert "value" not in result


def test_fixed_point_stability_is_read_from_records_not_coordinate_arrays(diagnostics) -> None:
    system = ts.systems.Logistic(r=2.0)
    entry = SimpleNamespace(family="map")
    card = diagnostics._equilibria(entry, system, forced=False)
    assert card["value"] == 2
    assert card["stable"] == 1
    assert card["unstable"] == 1
    assert "not a completeness proof" in card["note"]


def test_exact_symbolic_parameter_constants_have_serializable_numeric_values(diagnostics) -> None:
    import symengine

    parameters = {"delta": symengine.pi * 2 / 3, "a": np.float64(0.2)}
    converted = diagnostics._jsonable(parameters)
    assert converted["delta"] == pytest.approx(2 * np.pi / 3)
    json.dumps(converted, allow_nan=False)


def test_numerical_fingerprint_ignores_prose_but_tracks_equations(diagnostics) -> None:
    original = 'class Model:\n """Old explanation."""\n _reference="Old citation"\n def f(self,x):\n  return 2*x\n'
    editorial = original.replace("Old explanation", "New explanation").replace(
        "Old citation", "New citation"
    )
    numerical = original.replace("2*x", "3*x")
    assert diagnostics._python_semantics(original) == diagnostics._python_semantics(editorial)
    assert diagnostics._python_semantics(original) != diagnostics._python_semantics(numerical)


def test_fingerprint_is_portable_between_checkout_and_wheel_paths_and_newlines(
    diagnostics, tmp_path
) -> None:
    checkout = tmp_path / "checkout.py"
    wheel = tmp_path / "installed.py"
    checkout.write_bytes(b"def f(x):\n return x + 1\n")
    wheel.write_bytes(b"def f(x):\r\n return x + 1\r\n")

    def signature(path):
        return (("package/model.py", str(path), path.stat().st_mtime_ns, path.stat().st_size),)

    assert diagnostics._implementation_digest(
        signature(checkout)
    ) == diagnostics._implementation_digest(signature(wheel))


def test_a_parent_equation_change_invalidates_a_catalogue_subclass(
    diagnostics, monkeypatch
) -> None:
    child = registry.get("Coullet")
    original = diagnostics.fingerprint(child)
    source_digest = diagnostics._class_digest

    def changed(cls):
        return b"changed inherited equations" if cls.__name__ == "Arneodo" else source_digest(cls)

    monkeypatch.setattr(diagnostics, "_class_digest", changed)
    assert diagnostics.fingerprint(child) != original


def test_disabled_compute_uses_matching_recorded_evidence(diagnostics, monkeypatch) -> None:
    entry = next(e for e in registry.all_systems() if e.name == "Lorenz")
    record = {"name": "Lorenz", "fingerprint": diagnostics.fingerprint(entry), "sentinel": 7}
    monkeypatch.setattr(diagnostics, "load_records", lambda: {"Lorenz": record})
    monkeypatch.setenv("TSD_DOCS_PROPS", "0")
    result = diagnostics.compute_diagnostics(entry)
    assert result["sentinel"] == record["sentinel"]
    assert result["reference"] == diagnostics.reference(entry)


def test_stale_record_cannot_be_presented_as_current(diagnostics, monkeypatch) -> None:
    entry = next(e for e in registry.all_systems() if e.name == "Lorenz")
    monkeypatch.setattr(diagnostics, "load_records", lambda: {"Lorenz": {"fingerprint": "old"}})
    monkeypatch.setenv("TSD_DOCS_PROPS", "0")
    result = diagnostics.compute_diagnostics(entry)
    assert result["lyapunov_spectrum"]["status"] == "unavailable"
    assert "value" not in result["lyapunov_spectrum"]


def test_timeout_retains_a_completed_root_search(diagnostics, monkeypatch) -> None:
    import subprocess

    partial = {
        "name": "Lorenz",
        "equilibria": {"status": "measured", "value": 3},
    }

    def interrupted(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1.0, output=json.dumps(partial).encode())

    monkeypatch.setattr(diagnostics.subprocess, "run", interrupted)
    entry = next(e for e in registry.all_systems() if e.name == "Lorenz")
    record = diagnostics._in_subprocess(entry, timeout=1.0)
    assert record["equilibria"] == partial["equilibria"]
    assert record["lyapunov_spectrum"]["status"] == "unavailable"


def test_property_markup_escapes_evidence_and_explains_absent_answers(
    diagnostics, monkeypatch
) -> None:
    properties = importlib.import_module("properties")
    missing = diagnostics.unavailable("<unsafe> & unresolved")
    props = {
        "lyapunov_spectrum": missing,
        "kaplan_yorke": missing,
        "divergence": diagnostics.unavailable("discrete map", applicable=False),
        "equilibria": {"status": "measured", "value": 0, "settings": {"seed": 0}},
        "reference": {"source": "<unsafe reference>"},
        "provenance": {"parameters": {"name": "<unsafe parameter>"}},
    }
    markup = properties.to_markdown(props)
    assert "TODO" not in markup
    assert "Not applicable" in markup
    assert "none found in this search" in markup
    assert "<unsafe" not in markup
    assert "&lt;unsafe" in markup
    assert "Method and evidence" in markup
    json.dumps(props, allow_nan=False)
