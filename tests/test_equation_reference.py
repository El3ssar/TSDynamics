"""Correctness gate for the system-catalogue right-hand sides.

This module defends the catalogue against *transcription* and
*operator-precedence* bugs in the ``_equations`` / ``_step`` / ``_drift``
kernels — the class of defect that produced the ``WindmiReduced`` ``p**1/2``
bug (a ``p**(1/2)`` written without parentheses lowers to ``p / 2`` because
``**`` binds tighter than ``/``).  It works in two complementary layers.

1. **Curated analytic checks** (the load-bearing layer).  For a hand-picked set
   of well-known systems whose equations are unambiguous in the literature, the
   library's own RHS is evaluated at a fixed, non-trivial state and parameter
   set, and the result is compared against a value computed **independently** —
   the equations written out by hand in this file from the cited reference.
   This is a genuine re-derivation: the expected numbers come from plain Python
   arithmetic in the test, never from the same lowering / evaluator code path,
   so a tautology is impossible and the assertion fails the instant a kernel's
   math drifts from the cited form.

2. **Drift snapshot** (the long-tail layer).  Every catalogue system (all 151
   today) is lowered to its engine IR tape, and a SHA-256 of the canonical
   string form of that tape is pinned in a committed golden file.  Any
   accidental edit to a kernel changes its lowered tape, flips the hash, and the
   snapshot test names the offending system.  Hashing keeps the golden file tiny
   (one short line per system) while still pinning every byte of every tape.
   The catalogue is correct as of wave 1, so the golden records the *current*
   (correct) tape hashes.

Regenerating the golden snapshot (only after a *deliberate*, reviewed change to
a kernel or to the lowering itself) is a one-liner::

    PYTHONPATH=src python -m tests.test_equation_reference --regenerate

Both layers stay in the fast tier: the RHS is evaluated at a single point and
each system is lowered once — nothing is integrated.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tsdynamics import registry
from tsdynamics.engine import compile as _compile

# --------------------------------------------------------------------------- #
# Layer 1 — curated independent analytic checks
# --------------------------------------------------------------------------- #
#
# Each entry re-derives the RHS by hand from the cited equations and compares it
# to the catalogue kernel evaluated at the same point.  The expected vector is
# built from plain Python arithmetic so the comparison is genuinely independent
# of the lowering / evaluator under test.
#
# State and parameters are deliberately non-trivial (no zeros that would mask a
# dropped term, no ones that would mask a missing coefficient).


def _lorenz_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Lorenz (1963): x'=σ(y−x), y'=ρx−xz−y, z'=xy−βz."""
    x, y, z = u
    s, r, b = p["sigma"], p["rho"], p["beta"]
    return [s * (y - x), r * x - x * z - y, x * y - b * z]


def _rossler_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Rössler (1976): x'=−y−z, y'=x+ay, z'=b+z(x−c)."""
    x, y, z = u
    a, b, c = p["a"], p["b"], p["c"]
    return [-y - z, x + a * y, b + z * (x - c)]


def _chen_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Chen & Ueta (1999): x'=a(y−x), y'=(c−a)x−xz+cy, z'=xy−bz."""
    x, y, z = u
    a, b, c = p["a"], p["b"], p["c"]
    return [a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z]


def _thomas_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Thomas: x'=−ax+b sin y, y'=−ay+b sin z, z'=−az+b sin x."""
    x, y, z = u
    a, b = p["a"], p["b"]
    return [
        -a * x + b * math.sin(y),
        -a * y + b * math.sin(z),
        -a * z + b * math.sin(x),
    ]


def _halvorsen_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Halvorsen: cyclic x'=−ax−by−bz−y² (and permutations)."""
    x, y, z = u
    a, b = p["a"], p["b"]
    return [
        -a * x - b * y - b * z - y**2,
        -a * y - b * z - b * x - z**2,
        -a * z - b * x - b * y - x**2,
    ]


def _duffing_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Forced Duffing (autonomous form): x'=y, y'=−δy−αx−βx³+γcos z, z'=ω."""
    x, y, z = u
    alpha, beta, delta, gamma, omega = (p["alpha"], p["beta"], p["delta"], p["gamma"], p["omega"])
    return [y, -delta * y - alpha * x - beta * x**3 + gamma * math.cos(z), omega]


def _forced_vdp_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Forced van der Pol: x'=y, y'=μ(1−x²)y−x+a sin z, z'=w."""
    x, y, z = u
    a, mu, w = p["a"], p["mu"], p["w"]
    return [y, mu * (1 - x**2) * y - x + a * math.sin(z), w]


def _forced_fhn_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Forced FitzHugh–Nagumo: v'=v−v³/3−w+I+f sin z, w'=γ(v+a−bw), z'=ω."""
    v, w, z = u
    a, b, curr, f, gamma, omega = (
        p["a"],
        p["b"],
        p["curr"],
        p["f"],
        p["gamma"],
        p["omega"],
    )
    return [v - v**3 / 3 - w + curr + f * math.sin(z), gamma * (v + a - b * w), omega]


def _windmi_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Reduced WINDMI (Horton 2001) — the system that carried the ``p**1/2`` bug.

    i' = a1(vsw − v)
    v' = b1·i − b2·|p|^(1/2) − b3·v
    p' = vsw² − |p|^(5/4)·vsw^(1/2)·(1 + tanh(z_clamped))/2

    The fractional powers are written with explicit parentheses here; a kernel
    that wrote ``**1/2`` (precedence bug → ``/ 2``) gives a different ``v'`` and
    ``p'`` and this check fails.  Two curated states are used: one where the
    ``tanh`` gate saturates (isolates the ``v'`` ``**(1/2)`` term) and one near
    ``i = 1`` where the gate is strictly interior, so the ``p'`` ``**(5/4)``
    term is genuinely exercised.
    """
    i, v, pp = u
    a1, b1, b2, b3, d1, vsw = (p["a1"], p["b1"], p["b2"], p["b3"], p["d1"], p["vsw"])
    clamp = 25.0  # WindmiReduced._TANH_CLAMP — invisible to the dynamics on-orbit
    z = d1 * (i - 1)
    z_clamped = (abs(z + clamp) - abs(z - clamp)) / 2
    idot = a1 * (vsw - v)
    vdot = b1 * i - b2 * abs(pp) ** (1 / 2) - b3 * v
    pdot = vsw**2 - abs(pp) ** (5 / 4) * vsw ** (1 / 2) * (1 + math.tanh(z_clamped)) / 2
    return [idot, vdot, pdot]


def _henon_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Hénon (1976): x'=1−ax²+y, y'=bx."""
    x, y = u
    a, b = p["a"], p["b"]
    return [1.0 - a * x**2 + y, b * x]


def _logistic_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Logistic (May 1976): x'=rx(1−x)."""
    (x,) = u
    return [p["r"] * x * (1 - x)]


def _ikeda_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Ikeda (1979): t=a−b/(1+x²+y²); x'=1+u(x cos t−y sin t); y'=u(x sin t+y cos t)."""
    x, y = u
    a, b, uu = p["a"], p["b"], p["u"]
    t = a - b / (1 + x**2 + y**2)
    return [
        1 + uu * (x * math.cos(t) - y * math.sin(t)),
        uu * (x * math.sin(t) + y * math.cos(t)),
    ]


def _chirikov_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Chirikov standard map: p'=p+k sin x, x'=x+p'."""
    pv, x = u
    pp = pv + p["k"] * math.sin(x)
    return [pp, x + pp]


def _tinkerbell_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Tinkerbell: x'=x²−y²+ax+by, y'=2xy+cx+dy."""
    x, y = u
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    return [x**2 - y**2 + a * x + b * y, 2 * x * y + c * x + d * y]


def _ulam_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Ulam: x'=a−bx²."""
    (x,) = u
    return [p["a"] - p["b"] * x**2]


def _ricker_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Ricker (1954): x'=x·exp(a−x)."""
    (x,) = u
    return [x * math.exp(p["a"] - x)]


def _folded_towel_expected(u: list[float], p: dict[str, float]) -> list[float]:
    """Rössler folded-towel map (3-D hyperchaotic)."""
    x, y, z = u
    a, b, c, d, e, f, g = (p["a"], p["b"], p["c"], p["d"], p["e"], p["f"], p["g"])
    return [
        a * x * (1 - x) - b * (y + c) * (1 - 2 * z),
        d * ((y + c) * (1 + 2 * z) - 1) * (1 - e * x),
        f * z * (1 - z) + g * y,
    ]


#: ``(case_id, system_name, state, override-params, expected-fn)``.  ``None``
#: params use the catalogue defaults.  A system may appear more than once (a
#: distinct ``case_id``) to exercise different terms / a parameter override.
#: States are chosen to exercise every term (no zeros/ones that mask a defect).
CASES: list[tuple[str, str, list[float], dict[str, float] | None, Any]] = [
    # ODEs ----------------------------------------------------------------
    ("Lorenz", "Lorenz", [2.0, -1.0, 0.5], None, _lorenz_expected),
    ("Rossler", "Rossler", [0.7, -0.3, 1.2], None, _rossler_expected),
    ("Chen", "Chen", [1.5, -0.5, 2.0], None, _chen_expected),
    ("Thomas", "Thomas", [0.6, 1.1, -0.4], None, _thomas_expected),
    ("Halvorsen", "Halvorsen", [-1.0, 0.5, 0.3], None, _halvorsen_expected),
    ("Duffing", "Duffing", [0.4, -0.7, 1.3], None, _duffing_expected),
    ("ForcedVanDerPol", "ForcedVanDerPol", [0.3, 1.2, 0.8], None, _forced_vdp_expected),
    ("ForcedFitzHughNagumo", "ForcedFitzHughNagumo", [0.5, 0.2, 1.1], None, _forced_fhn_expected),
    # WindmiReduced twice: a saturated-gate state (isolates v' **(1/2)) and a
    # near-i=1 state with a param override (interior gate exercises p' **(5/4)
    # and covers the parameter-override path).
    ("WindmiReduced[gate=sat]", "WindmiReduced", [0.5, 0.3, 2.0], None, _windmi_expected),
    (
        "WindmiReduced[gate=interior]",
        "WindmiReduced",
        [1.001, 0.3, 2.0],
        {"b2": 0.1},
        _windmi_expected,
    ),
    # Maps ----------------------------------------------------------------
    ("Henon", "Henon", [0.3, 0.4], None, _henon_expected),
    ("Logistic", "Logistic", [0.6], None, _logistic_expected),
    ("Ikeda", "Ikeda", [0.7, -0.5], None, _ikeda_expected),
    ("Chirikov", "Chirikov", [0.4, 1.3], None, _chirikov_expected),
    ("Tinkerbell", "Tinkerbell", [-0.3, 0.5], None, _tinkerbell_expected),
    ("Ulam", "Ulam", [0.35], None, _ulam_expected),
    ("Ricker", "Ricker", [0.8], None, _ricker_expected),
    ("FoldedTowel", "FoldedTowel", [0.4, 0.2, 0.5], None, _folded_towel_expected),
]


def _evaluate_rhs(entry: Any, state: list[float], params: dict[str, float] | None) -> np.ndarray:
    """Evaluate the catalogue kernel's RHS at ``state`` via the library's own path.

    ODEs go through :meth:`ContinuousSystem._rhs_numeric` (the SymEngine-Lambdified
    numeric RHS); maps call ``_step`` directly with the parameters in declared
    order.  Neither path is the one the test's expected value uses, so the
    comparison is independent.
    """
    cls = entry.cls
    system = cls() if params is None else cls(params=params)
    u = np.asarray(state, dtype=float)
    if entry.family == "map":
        param_values = [system.params[k] for k in cls.params]
        out = cls._step(u if cls.dim > 1 else u[0], *param_values)
        return np.asarray(out, dtype=float).ravel()
    if entry.family == "ode":
        return np.asarray(system._rhs_numeric()(u, 0.0), dtype=float).ravel()
    raise AssertionError(f"curated check does not support family {entry.family!r}")


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_curated_rhs_matches_independent_derivation(
    case: tuple[str, str, list[float], dict[str, float] | None, Any],
) -> None:
    """Catalogue RHS equals a by-hand re-derivation of the cited equations.

    Fails if a kernel's math drifts from the literature form — including the
    ``WindmiReduced`` ``p**(1/2)`` precedence bug (a ``**1/2`` kernel returns a
    different ``v'``/``p'`` than this hand-derived expected vector).
    """
    case_id, name, state, params, expected_fn = case
    entry = registry.get(name)
    assert entry is not None, f"curated system {name!r} is not in the registry"

    effective_params = dict(entry.params)
    if params is not None:
        effective_params.update(params)

    actual = _evaluate_rhs(entry, state, params)
    expected = np.asarray(expected_fn(state, effective_params), dtype=float)

    assert actual.shape == expected.shape, (
        f"{case_id}: RHS returned shape {actual.shape}, expected {expected.shape}"
    )
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        diff = actual - expected
        raise AssertionError(
            f"{case_id}: catalogue RHS disagrees with the independent re-derivation.\n"
            f"  state    = {state}\n"
            f"  expected = {expected.tolist()}\n"
            f"  actual   = {actual.tolist()}\n"
            f"  diff     = {diff.tolist()}"
        )


def test_curated_set_covers_both_families_and_windmi() -> None:
    """Guard the curated set: it must cover ODEs *and* maps and pin WindmiReduced.

    Keeps the layer honest if someone trims the table — a curated set with no
    maps (or no WindmiReduced) would silently stop guarding those.
    """
    names = {c[1] for c in CASES}
    families = {registry.get(n).family for n in names}
    assert {"ode", "map"} <= families, f"curated set must span ode+map, got {families}"
    assert "WindmiReduced" in names, "the regression system must stay curated"
    # The interior-gate WindmiReduced case must be present so the p' **(5/4)
    # term is genuinely exercised (the saturated-gate state zeroes it out).
    assert any(c[0] == "WindmiReduced[gate=interior]" for c in CASES), (
        "WindmiReduced needs an interior-gate case to exercise the p' **(5/4) term"
    )


# --------------------------------------------------------------------------- #
# Layer 2 — drift snapshot over the whole catalogue
# --------------------------------------------------------------------------- #

GOLDEN_PATH = Path(__file__).with_name("_equation_reference_golden.txt")


def _tape_for(entry: Any) -> Any:
    """Lower a catalogue system to its engine IR tape (drift tape for an SDE)."""
    system = entry.cls()
    family = entry.family
    if family == "map":
        return _compile.lower_map(system)
    if family == "dde":
        tape, _slots = _compile.lower_dde(system)
        return tape
    if family == "sde":
        return _compile.lower_sde(system).drift
    return _compile.lower_ode(system)


def _canonical_tape(tape: Any) -> str:
    """Serialise a tape to a stable one-line string.

    The string captures the opcode stream, the register wiring (``a``/``b``),
    the immediate pool, the output registers and the state/param counts — i.e.
    everything that changes when a kernel's math changes (operator precedence,
    a dropped term, a swapped coefficient).  Integers print as ``int``; floats
    print at full ``float64`` precision so a changed constant is caught.
    """

    def ints(arr: Any) -> str:
        return ",".join(str(int(v)) for v in np.asarray(arr).ravel().tolist())

    def floats(arr: Any) -> str:
        return ",".join(format(float(v), ".17g") for v in np.asarray(arr).ravel().tolist())

    return "|".join(
        [
            f"ops={ints(tape.ops)}",
            f"a={ints(tape.a)}",
            f"b={ints(tape.b)}",
            f"imm={floats(tape.imm)}",
            f"out={ints(tape.outputs)}",
            f"n_state={int(tape.n_state)}",
            f"n_param={int(tape.n_param)}",
        ]
    )


def _tape_hash(tape: Any) -> str:
    """SHA-256 of the canonical tape string (pins every byte; tiny to store)."""
    return hashlib.sha256(_canonical_tape(tape).encode("utf-8")).hexdigest()


def _build_snapshot() -> dict[str, str]:
    """Map every catalogue system name to its canonical-tape hash."""
    return {entry.name: _tape_hash(_tape_for(entry)) for entry in registry.all_systems()}


def _serialize_snapshot(snapshot: dict[str, str]) -> str:
    """Render the snapshot to the golden-file text (one ``name\\thash`` per line)."""
    header = (
        "# Equation-reference drift snapshot — SHA-256 of the canonical IR tape\n"
        "# per catalogue system. Regenerate ONLY after a deliberate, reviewed\n"
        "# change to a kernel or to the lowering:\n"
        "#   PYTHONPATH=src python -m tests.test_equation_reference --regenerate\n"
        "# Format: <SystemName>\\t<sha256-of-canonical-tape>\n"
    )
    lines = [f"{name}\t{snapshot[name]}" for name in sorted(snapshot)]
    return header + "\n".join(lines) + "\n"


def _load_golden() -> dict[str, str]:
    """Parse the committed golden file into ``name -> tape-hash``."""
    text = GOLDEN_PATH.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, canon = line.partition("\t")
        out[name] = canon
    return out


def test_golden_snapshot_exists() -> None:
    """The committed golden file must exist (regenerate it if this fails)."""
    assert GOLDEN_PATH.exists(), (
        f"missing golden snapshot {GOLDEN_PATH.name!r}; regenerate with "
        "`PYTHONPATH=src python -m tests.test_equation_reference --regenerate`"
    )


def test_catalogue_tapes_match_snapshot() -> None:
    """Every catalogue RHS lowers to its pinned tape hash — the drift gate.

    Catches the long tail of transcription bugs across all 151 systems: any
    accidental edit to an ``_equations`` / ``_step`` / ``_drift`` body changes
    its lowered tape, flips the SHA-256, and this test names the offending
    system(s).
    """
    current = _build_snapshot()
    golden = _load_golden()

    current_names = set(current)
    golden_names = set(golden)

    new = sorted(current_names - golden_names)
    removed = sorted(golden_names - current_names)
    changed = sorted(n for n in current_names & golden_names if current[n] != golden[n])

    if new or removed or changed:
        lines: list[str] = ["catalogue RHS snapshot mismatch:"]
        if new:
            lines.append(
                f"  NEW systems (not in golden): {new}\n"
                "    -> if intentional, regenerate the golden snapshot."
            )
        if removed:
            lines.append(f"  REMOVED systems (in golden, not in catalogue): {removed}")
        for n in changed:
            lines.append(
                f"  CHANGED: {n} (tape hash {golden[n][:12]}… -> {current[n][:12]}…) — "
                "its lowered RHS changed; if deliberate, regenerate."
            )
        lines.append(
            "If a change is deliberate, regenerate with "
            "`PYTHONPATH=src python -m tests.test_equation_reference --regenerate`."
        )
        raise AssertionError("\n".join(lines))


def test_snapshot_covers_full_catalogue() -> None:
    """The snapshot pins the *entire* live catalogue (no system slips the gate)."""
    n_systems = len(list(registry.all_systems()))
    assert n_systems == len(_build_snapshot()), "snapshot lost a catalogue system"
    assert n_systems >= 151, f"catalogue shrank unexpectedly to {n_systems} systems"


def _regenerate() -> None:
    """Write the golden snapshot from the current (assumed-correct) catalogue."""
    snapshot = _build_snapshot()
    GOLDEN_PATH.write_text(_serialize_snapshot(snapshot), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} ({len(snapshot)} systems)")


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print("pass --regenerate to rewrite the golden snapshot")
