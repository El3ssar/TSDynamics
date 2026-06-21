"""Signature-lint for the v4 calling convention (stream **WS-CONV**).

Enforces the frozen naming glossary (``docs/contributing/glossary.md``) over the
*public analysis/transform surface* — every function registered in
``registry.analyses`` / ``registry.transforms``:

1. its **first positional argument** is ``system`` or ``data`` (the two canonical
   input roles), unless the ``(function, first-arg)`` pair names a *prior result*
   on the §5 whitelist;
2. **no parameter** uses a banned spelling from glossary §2, unless the
   ``(function, parameter)`` pair is a §5 homonym carve-out.

This is the focused precursor the broader CI naming gate (``WS-NAMEGATE``) later
generalises; both read from the same glossary, so a banned spelling can never
re-enter a public signature.  The check is pure :func:`inspect.signature`
introspection — no engine, fast tier.
"""

from __future__ import annotations

import inspect

import pytest

import tsdynamics.transforms  # noqa: F401  (populates registry.transforms)
from tsdynamics import registry

# ── glossary §1: the two canonical first-argument roles ───────────────────────
CANONICAL_FIRST_ARGS = frozenset({"system", "data"})

# Banned first-arg spellings (glossary §1) — all collapse to system / data.
BANNED_FIRST_ARGS = frozenset(
    {"sys", "sys_or_traj", "map_sys", "observable", "source", "x", "series"}
)

# §5: prior-result first arguments, named by the kind of result they consume.
# Whitelisted as exact (function, first-arg) pairs.
PRIOR_RESULT_FIRST_ARG = {
    "kaplan_yorke_dimension": "spectrum",
    "uncertainty_exponent": "basins",
    "wada_property": "basins",
    "basin_entropy": "basins",
    "resilience": "result",
    "tipping_points": "result",
}

# ── glossary §2: banned parameter spellings → their canonical concept ──────────
# Built straight from the §2 "Bans" column.  A parameter whose name is a key here
# is rejected unless its (function, parameter) pair is on the §5 whitelist.
BANNED_PARAMS = {
    # initial condition → ic
    "x0": "ic",
    "initial": "ic",
    "u0": "ic",
    "y0": "ic",
    # RNG seed → seed
    "random_state": "seed",
    "rng": "seed",
    # discard-transient → transient
    "burn_in": "transient",
    "n_transient": "transient",
    "warmup": "transient",
    # integration horizon → final_time
    "t_final": "final_time",
    "tmax": "final_time",
    # iteration horizon → n
    "steps": "n",
    "n_rescale": "n",
    # step size → dt
    "h": "dt",
    # observed component → component
    "components": "component",
    "observable": "component",
    "coord": "component",
    "col": "component",
    # embedding dimension → dimension
    "m": "dimension",
    "emb_dim": "dimension",
    "dim": "dimension",
    # embedding delay → delay
    "tau": "delay",
    "lag": "delay",
    "max_lag": "max_delay",
    # Theiler window → theiler
    "theiler_window": "theiler",
    "w": "theiler",
    # nearest-neighbour count → n_neighbors
    "min_neighbors": "n_neighbors",
    "num_neighbors": "n_neighbors",
    # spatial region → region
    "grid": "region",
    "box": "region",
    "domain": "region",
    "bounds": "region",
}

# §5 homonym carve-outs: exact (function, parameter) pairs that *may* use a token
# which is banned elsewhere.  (None of the canonical homonyms — k, k_max, step,
# horizon, max_steps, max_delay, fs — are in BANNED_PARAMS, so this stays empty
# today; it is kept as the documented extension point.)
HOMONYM_WHITELIST: frozenset[tuple[str, str]] = frozenset()


def _registered() -> list[tuple[str, object]]:
    """Every registered analysis + transform as ``(name, callable)`` pairs."""
    pairs: list[tuple[str, object]] = []
    for reg in (registry.analyses, registry.transforms):
        for entry in reg.all():
            pairs.append((entry.name, entry.obj))
    return pairs


def _params(fn: object) -> list[inspect.Parameter]:
    return list(inspect.signature(fn).parameters.values())


_REGISTERED = _registered()


@pytest.mark.parametrize("name,fn", _REGISTERED, ids=[n for n, _ in _REGISTERED])
def test_first_argument_is_canonical(name: str, fn: object) -> None:
    """The first positional argument is ``system`` / ``data`` (or a §5 prior-result)."""
    positional = [p for p in _params(fn) if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if not positional:  # a pure keyword-only consumer (rare) has no first-arg role
        return
    first = positional[0].name
    if name in PRIOR_RESULT_FIRST_ARG:
        assert first == PRIOR_RESULT_FIRST_ARG[name], (
            f"{name}: prior-result first arg should be "
            f"{PRIOR_RESULT_FIRST_ARG[name]!r}, got {first!r}."
        )
        return
    assert first not in BANNED_FIRST_ARGS, (
        f"{name}: first argument {first!r} is a banned spelling — use 'system' "
        f"(a System) or 'data' (a measured series)."
    )
    assert first in CANONICAL_FIRST_ARGS, (
        f"{name}: first argument {first!r} is neither 'system' nor 'data' (and is "
        f"not a whitelisted prior-result consumer)."
    )


@pytest.mark.parametrize("name,fn", _REGISTERED, ids=[n for n, _ in _REGISTERED])
def test_no_banned_parameter_spellings(name: str, fn: object) -> None:
    """No parameter uses a glossary §2 banned spelling (outside §5 carve-outs)."""
    offenders = []
    for p in _params(fn):
        if p.name in BANNED_PARAMS and (name, p.name) not in HOMONYM_WHITELIST:
            offenders.append(f"{p.name!r} (use {BANNED_PARAMS[p.name]!r})")
    assert not offenders, f"{name}: banned parameter spelling(s): {', '.join(offenders)}."


def test_headline_first_args_are_system_or_data() -> None:
    """Spot-check the worked-example offenders from the dossier are fixed."""
    fixed = {
        "lyapunov_spectrum": "system",
        "zero_one_test": "system",
        "poincare_section": "system",
        "return_map": "system",
        "orbit_diagram": "system",
        "periodic_orbits": "system",
        "max_lyapunov": "system",
        "lyapunov_from_data": "data",
    }
    by_name = dict(_REGISTERED)
    for fn_name, expected in fixed.items():
        fn = by_name[fn_name]
        first = next(
            p.name for p in _params(fn) if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        assert first == expected, f"{fn_name}: first arg {first!r} != {expected!r}."


def test_region_and_seed_additions() -> None:
    """``ts.data.region`` exists and seed= reached the four sweep entry points."""
    from tsdynamics import data

    g = data.region([(-1.0, 1.0, 4), (-1.0, 1.0, 4)])
    assert tuple(g.shape) == (4, 4)

    by_name = dict(_REGISTERED)
    for fn_name in ("orbit_diagram", "poincare_section", "return_map", "basins_of_attraction"):
        params = {p.name for p in _params(by_name[fn_name])}
        assert "seed" in params, f"{fn_name} is missing the seed= keyword."
