"""Render auditable numerical property cards for every catalogue system.

The measured spectrum, its Kaplan–Yorke estimate, and roots found by a seeded
search come from :mod:`diagnostics` and its versioned dataset. Reference values
retain their own parameters and never silently replace a default-parameter
measurement. The fourth card is a symbolic Jacobian trace, with explicit caveats
for delay histories, stochastic drift, and discrete maps.

``TSD_DOCS_PROPS=0`` permits matching recorded measurements but disables missing
recomputations. Unavailable and inapplicable answers state their reasons. A root
search reports roots *found*, never a completeness claim.
"""

from __future__ import annotations

import functools
import hashlib
import html
import importlib.util
import inspect
import json
import os
import pathlib
import subprocess
import sys
import warnings
from typing import Any

import numpy as np


@functools.lru_cache(maxsize=1)
def _diagnostics_module():
    """Load the sibling when MkDocs imports this file directly by its path."""
    path = pathlib.Path(__file__).with_name("diagnostics.py")
    spec = importlib.util.spec_from_file_location("_tsdynamics_docs_diagnostics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load catalogue diagnostics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quiet_numerics(fn):
    """Silence the expected FP / divergence ``RuntimeWarning``s of build-time work.

    Computing the property cards runs a bounded Lyapunov QR and a multi-start /
    box fixed-point search over each catalogue system; random ICs and wide search
    boxes legitimately overflow or go non-finite, and the estimators already
    handle that (retry, drop non-finite, or record an unavailable result). This keeps the ``--strict``
    docs-build log clean without changing any card — only the noisy warning is
    suppressed for the duration of the computation.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(*args, **kwargs)

    return wrapper


ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-props"

#: Bump when the computed-quantity logic or card shaping materially changes
#: (cache buster — invalidates every on-disk props payload).
PROPS_VERSION = "4"

# A new model must not turn an ordinary docs build into an unbounded analysis
# job. Existing provenance snapshots bypass this worker; fresh results are cached.
NEW_MODEL_TIMEOUT = 30.0
WORKER_PATH = pathlib.Path(__file__).with_name("diagnostics_worker.py")

#: The ∇·f trace is *collapsed* to a compact summary (rather than printed in
#: full) when the system is a method-of-lines field, its effective state
#: dimension exceeds :data:`_DIV_MAX_DIM`, or its rendered LaTeX exceeds
#: :data:`_DIV_MAX_LATEX_LEN` characters.  Rendered-LaTeX length is the reliable
#: bloat signal: a field (Gray–Scott ≈ 84 000 chars) or a low-dim system with a
#: pathological rational trace (Glycolytic oscillation ≈ 50 000, Excitable cell
#: ≈ 88 000) wrecks page load, while a tidy low-dim ODE (Lorenz ``-β-σ-1``,
#: Rössler ``a-c+x``, and rich-but-readable ones like Lorenz-bounded at ≈ 370
#: chars) stays well under the limit and renders in full.
_DIV_MAX_DIM = 8
_DIV_MAX_LATEX_LEN = 500


# --------------------------------------------------------------------------- #
# Cache plumbing
# --------------------------------------------------------------------------- #
def cache_key(entry) -> str:
    """Content hash: class source + default params + module version + compute flag.

    Include estimator/protocol fingerprints and recorded evidence. A preview
    without current evidence cannot masquerade as a completed measurement.
    """
    cls_src = inspect.getsource(entry.cls)
    params = repr(sorted((entry.params or {}).items()))
    flag = "compute" if os.environ.get("TSD_DOCS_PROPS", "1") != "0" else "preview"
    diagnostics = _diagnostics_module()

    recorded = diagnostics.load_records().get(entry.name, {})
    evidence = json.dumps(recorded, sort_keys=True)
    reference = json.dumps(diagnostics.reference(entry), sort_keys=True)
    identity = f"{entry.cls.__module__}:{entry.cls.__qualname__}"
    blob = (
        cls_src
        + identity
        + WORKER_PATH.read_text()
        + params
        + PROPS_VERSION
        + flag
        + diagnostics.fingerprint(entry)
        + evidence
        + reference
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _cache_path(entry) -> pathlib.Path:
    return CACHE_DIR / f"{entry.name}-{cache_key(entry)}.json"


# --------------------------------------------------------------------------- #
# Sentinels
# --------------------------------------------------------------------------- #
def _unavailable(reason: str) -> dict[str, Any]:
    """Explain an absent symbolic answer without inventing a measurement."""
    return {"status": "unavailable", "reason": reason}


def _ok(value: Any, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "ok", "value": value}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# Lyapunov spectrum + Kaplan–Yorke
# --------------------------------------------------------------------------- #
def _known_lyapunov(entry) -> dict[str, Any] | None:
    """Read reference metadata with its own conditions, separately from measurements."""
    return _diagnostics_module().reference(entry)


def _compute_lyapunov(entry) -> dict[str, Any]:
    """Read the current, reproducible finite-time measurement for this system."""
    return _compute_diagnostics(entry)["lyapunov_spectrum"]


def _compute_diagnostics(entry) -> dict[str, Any]:
    """Reuse provenance snapshots, or measure the exact importable model once."""
    diagnostics = _diagnostics_module()
    saved = diagnostics.load_records().get(entry.name)

    def absent(reason):
        return {
            "name": entry.name,
            "family": entry.family,
            "reference": diagnostics.reference(entry),
            **{
                key: diagnostics.unavailable(reason)
                for key in ("lyapunov_spectrum", "kaplan_yorke", "equilibria")
            },
        }

    module = entry.cls.__module__
    qualname = entry.cls.__qualname__
    try:
        source_file = inspect.getsourcefile(entry.cls)
    except (TypeError, OSError):
        source_file = None
    if (
        module == "__main__"
        or "<locals>" in qualname
        or not source_file
        or not pathlib.Path(source_file).is_file()
    ):
        return absent(
            "This class exists only in the current session. Put it in an importable "
            "module to compute reproducible diagnostics in the bounded worker."
        )
    fingerprint = diagnostics.fingerprint(entry)
    if saved is not None and saved.get("fingerprint") == fingerprint:
        return diagnostics.compute_diagnostics(entry)
    if os.environ.get("TSD_DOCS_PROPS", "1") == "0":
        return diagnostics.compute_diagnostics(entry)
    request = {
        "module": module,
        "qualname": qualname,
        "source_file": str(pathlib.Path(source_file).resolve()),
        "family": entry.family,
        "fingerprint": fingerprint,
    }
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        str(pathlib.Path(path or os.getcwd()).resolve()) for path in sys.path
    )
    try:
        result = subprocess.run(
            [sys.executable, str(WORKER_PATH)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=environment,
            timeout=NEW_MODEL_TIMEOUT,
        )
        record = diagnostics._last_record(result.stdout)
        if result.returncode or record is None:
            message = result.stderr.strip().splitlines()
            reason = "The numerical worker could not measure the requested model: " + (
                message[-1] if message else "no completed diagnostic record."
            )
            if record is None:
                return absent(reason)
            record["lyapunov_spectrum"] = diagnostics.unavailable(reason)
            record["kaplan_yorke"] = diagnostics.unavailable(reason)
    except subprocess.TimeoutExpired as exc:
        reason = f"The new-model computation budget of {NEW_MODEL_TIMEOUT:g}s was exceeded."
        record = diagnostics._last_record(exc.stdout)
        if record is None:
            return absent(reason)
        record["lyapunov_spectrum"] = diagnostics.unavailable(reason)
        record["kaplan_yorke"] = diagnostics.unavailable(reason)
    if (
        record.get("name") != entry.name
        or record.get("family") != entry.family
        or record.get("fingerprint") != fingerprint
    ):
        return absent("The numerical worker returned an unverified model identity.")
    record.setdefault("provenance", {})["model_identity"] = {
        "module": module,
        "class": qualname,
        "worker_budget_seconds": NEW_MODEL_TIMEOUT,
    }
    return record


def _lyapunov_card(entry) -> dict[str, Any]:
    """Read measurements separately from any published reference conditions."""
    return _compute_lyapunov(entry)


def _kaplan_yorke_card(lyap: dict[str, Any]) -> dict[str, Any]:
    """Derive a dimension only when the recorded leading spectrum closes."""
    diagnostics = _diagnostics_module()

    return diagnostics.kaplan_yorke(lyap)


# --------------------------------------------------------------------------- #
# Phase-space divergence ∇·f  (symbolic trace of the Jacobian)
# --------------------------------------------------------------------------- #
def _state_names(cls, dim: int) -> list[str]:
    """Component names (``variables`` ClassVar) or ``y_i`` placeholders.

    Mirrors :mod:`equations` so the divergence LaTeX uses the same symbols as
    the rendered equations (``x``/``y``/``z`` rather than ``y(0)``…).
    """
    names = getattr(cls, "variables", None)
    if names and len(names) == dim:
        return list(names)
    return [f"y_{{{i}}}" for i in range(dim)]


def _param_symbols(cls, sys_obj) -> dict[str, Any]:
    """Parameters as symengine symbols (structural params stay numeric)."""
    import symengine

    structural = getattr(cls, "_structural_params", frozenset())
    return {k: (v if k in structural else symengine.Symbol(k)) for k, v in sys_obj.params.items()}


def _divergence_expr(entry):
    """Symbolic ∇·f = Σ ∂f_i/∂y_i for a flow (ODE) / drift (SDE) / DDE.

    Returns ``(div, state_syms, dim)`` — the expanded divergence, the set of
    instantaneous-state symbols it was differentiated against, and the state
    dimension.  Built from named state symbols (so the LaTeX matches the
    equations card).  For a DDE the delayed terms ``y(i, t-τ)`` are independent
    function symbols, so differentiating w.r.t. the instantaneous state gives the
    *instantaneous-state* trace, not the divergence of the history flow. Raises on any
    system whose RHS we cannot lower symbolically (variable-dim, NumPy bodies,
    …); the caller records an unavailable answer with its reason.
    """
    import symengine

    sys_obj = entry.cls()
    dim = sys_obj.dim
    if dim is None:
        raise ValueError("variable-dimensional system")

    names = _state_names(entry.cls, dim)
    syms = [symengine.Symbol(n) for n in names]
    t = symengine.Symbol("t")
    params = _param_symbols(entry.cls, sys_obj)

    if entry.family == "sde":

        def y(i):
            return syms[int(i)]

        exprs = list(entry.cls._drift(y, t, **params))
    elif entry.family == "dde":

        def y(i, time=None):  # type: ignore[misc]
            if time is None:
                return syms[int(i)]
            return symengine.Function(names[int(i)])(time)

        exprs = list(entry.cls._equations(y, t, **params))
    else:  # ode

        def y(i):
            return syms[int(i)]

        exprs = list(entry.cls._equations(y, t, **params))

    div = symengine.expand(sum(symengine.diff(exprs[i], syms[i]) for i in range(dim)))
    return div, set(syms), dim


def _divergence_card(entry) -> dict[str, Any]:
    """Divergence card: LaTeX of ∇·f for a flow; a qualitative note for a map.

    The symbolic trace is cheap, so it is computed regardless of
    ``TSD_DOCS_PROPS`` (it is not an integration).  A map has no continuous-flow
    divergence — the discrete analogue is the Jacobian *determinant* (the
    per-step phase-space volume factor), which we deliberately do not render as
    ``∇·f`` and instead mark as not-applicable with a pointer.

    A **method-of-lines field** (Gray–Scott, Swift–Hohenberg) or any high-dim
    flow whose ∇·f expands into a wall of near-identical per-node terms
    (Lorenz-96, Kuramoto–Sivashinsky) would print as an enormous sum that wrecks
    readability and page load.  Such a trace is *collapsed*
    (:func:`_collapse_divergence`) to a compact summary — the parameter-only
    constant part plus a "(N-cell field)" note, or a one-line "too large to
    display" — while a low-dim ODE keeps its full, tidy ∇·f.
    """
    if entry.family == "map":
        return {
            "status": "na",
            "reason": "discrete map — flow divergence undefined (per-step contraction is |det J|)",
        }
    try:
        div, state_syms, dim = _divergence_expr(entry)
    except Exception as exc:  # noqa: BLE001 — symbolic lowering failed
        return _unavailable(f"symbolic divergence unavailable: {type(exc).__name__}")
    if _has_nonsmooth_derivative(div):
        # A non-smooth (abs / sign / fractional-power) RHS leaves an unresolved
        # ``Derivative(abs(...))`` / ``sign'`` in the symbolic trace.  The library
        # resolves these a.e. at *runtime* (``_resolve_derivative_nodes``), but as a
        # static LaTeX card the raw form is misleading — report it as not rendered
        # rather than print a wrong-looking expression.
        return {
            "status": "na",
            "reason": "non-smooth right-hand side — divergence is piecewise "
            "(defined almost everywhere)",
        }
    try:
        import sympy

        latex = sympy.latex(div._sympy_())
    except Exception as exc:  # noqa: BLE001
        return _unavailable(f"divergence render failed: {type(exc).__name__}")

    constant = not (set(div.free_symbols) & state_syms)

    # Collapse an enormous field / high-dim / pathologically-long trace to a
    # compact summary.  A short, tidy ∇·f (a low-dim ODE) falls straight through
    # and renders in full.
    if _is_field(entry) or dim > _DIV_MAX_DIM or len(latex) > _DIV_MAX_LATEX_LEN:
        return _collapse_divergence(entry, div, state_syms, dim)

    return _ok(latex, constant=constant)


def _is_field(entry) -> bool:
    """Whether ``entry`` is a spatially-extended (method-of-lines) field.

    True when the catalogue record exposes ``is_field`` / a ``field_shape``, or
    the class declares the ``_field_shape`` ClassVar — mirrors the field
    detection used by the figure / viewer tooling.
    """
    if getattr(entry, "is_field", False):
        return True
    if getattr(entry, "field_shape", None):
        return True
    return getattr(entry.cls, "_field_shape", None) is not None


def _collapse_divergence(entry, div, state_syms, dim: int) -> dict[str, Any]:
    """Compact ∇·f card for a field / high-dim / pathologically-long trace.

    Splits ∇·f into its parameter-only *constant* part (the uniform
    contraction/expansion rate — short and meaningful) and the *state-dependent*
    remainder (the per-node nonlinear terms).  The framing depends on the reason
    the full trace is unwieldy:

    - **Spatial** (a method-of-lines field, or a high-dim flow whose ∇·f is a sum
      of near-identical per-node contributions — Gray–Scott, Kuramoto–Sivashinsky):
      the remainder is described as "per-cell nonlinear terms (over the N-cell
      field)" — an honest summary of a repetitive spatial sum.
    - **Bulky** (a low-dim ODE with a pathological rational ∇·f — Excitable cell,
      Glycolytic oscillation): there is no field, so the remainder is described
      neutrally as "state-dependent part omitted (too large to display)".

    In both cases, if even the constant part does not render short, the whole
    expression is omitted with a one-line note (``status: "na"``).
    """
    import symengine

    spatial = _is_field(entry) or dim > _DIV_MAX_DIM
    if spatial:
        n = _cell_count(entry, dim)
        unit = "cell" if _is_field(entry) else "node"
        span = f"the {n}-{unit} field" if _is_field(entry) else f"{n} nodes"
        remainder_note = f"+ per-{unit} nonlinear terms (over {span})"
        constant_note = f"uniform over {span}"
        omit_reason = "too large to display — method-of-lines field"
    else:
        remainder_note = "constant part shown — state-dependent remainder omitted (too large)"
        constant_note = "constant"
        omit_reason = "too large to display — pathological rational divergence"

    args = div.args if isinstance(div, symengine.Add) else (div,)
    const_terms = [a for a in args if not (set(a.free_symbols) & state_syms)]
    has_state = len(const_terms) != len(args)
    const_part = sum(const_terms, symengine.Integer(0))

    # A zero constant part carries no information — showing "∇·f = 0" beside a
    # "remainder omitted" note would read as a false ∇·f ≡ 0.  Omit instead.
    if const_part == 0:
        return {"status": "na", "reason": omit_reason}

    const_latex: str | None = None
    try:
        import sympy

        const_latex = sympy.latex(const_part._sympy_())
    except Exception:  # noqa: BLE001 — a missing/failing sympy degrades to the omit note
        const_latex = None

    if const_latex is not None and len(const_latex) <= _DIV_MAX_LATEX_LEN:
        return {
            "status": "collapsed",
            "value": const_latex,
            "constant": not has_state,
            "note": remainder_note if has_state else constant_note,
        }

    # Even the constant part is unwieldy — omit with a clear one-line note.
    return {"status": "na", "reason": omit_reason}


def _cell_count(entry, dim: int) -> int:
    """Return the field-cell / method-of-lines node count (the ``N`` in the note).

    Uses the declared ``field_shape`` cell count when available (a multi-block
    field's per-block grid size), else the state dimension.
    """
    shape = getattr(entry, "field_shape", None) or getattr(entry.cls, "_field_shape", None)
    if shape:
        import math

        return int(math.prod(shape))
    return int(dim)


def _has_nonsmooth_derivative(div) -> bool:
    """Whether ``div`` carries an unresolved non-smooth derivative artifact.

    A ``Derivative(abs(...))`` / ``sign'`` / ``Subs(...)`` left in the symbolic
    trace means the RHS is non-smooth (``abs``/``sign``/fractional power); such a
    divergence is piecewise and not a clean closed-form card.
    """
    text = str(div)
    return "Derivative" in text or "Subs" in text or "sign(" in text.lower()


# --------------------------------------------------------------------------- #
# Equilibria (best-effort)
# --------------------------------------------------------------------------- #
def _equilibria_card(entry) -> dict[str, Any]:
    """Read the found roots and their actual linear stability masks."""
    return _compute_diagnostics(entry)["equilibria"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@_quiet_numerics
def compute_properties(entry) -> dict[str, Any]:
    """Read recorded diagnostics and compute the symbolic divergence card.

    Every card carries a status, and measured cards retain settings, provenance
    and finite-time/search limitations. Preview mode reuses recorded evidence.
    """
    cached = _load_cache(entry)
    if cached is not None:
        return cached

    props = dict(_compute_diagnostics(entry))
    props["divergence"] = _divergence_card(entry)
    if entry.family == "dde":
        props["divergence"]["note"] = (
            "Instantaneous-state trace only; this is not the divergence of the infinite-dimensional history flow."
        )
    elif entry.family == "sde":
        props["divergence"]["note"] = (
            "Trace of the drift Jacobian; stochastic path contraction also depends on diffusion."
        )
    _store_cache(entry, props)
    return props


def _load_cache(entry) -> dict[str, Any] | None:
    try:
        path = _cache_path(entry)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — corrupt cache → recompute
        return None


def _store_cache(entry, props: dict[str, Any]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(entry).write_text(json.dumps(props, separators=(",", ":")), encoding="utf-8")
    except Exception:  # noqa: BLE001 — caching is best-effort
        pass


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _fmt_exponent(v: float) -> str:
    # Exact zero (the marginal exponent of a flow) reads better unsigned.
    if v == 0:
        return "0"
    return f"{v:+.4g}"


def _lyapunov_md(card: dict[str, Any]) -> str:
    status = card.get("status")
    if status in ("measured", "reference", "ok"):
        spectrum = card["value"]
        body = "$" + ",\\; ".join(_fmt_exponent(v) for v in spectrum) + "$"
        unit = card.get("unit")
        if unit:
            body += f'<br><span class="ts-prop-note">{html.escape(unit)}</span>'
        if card.get("complete") is False:
            body += '<br><span class="ts-prop-note">leading exponents only</span>'
        return body
    if status == "partial":
        k = card.get("n_positive", "?")
        return f"{k} positive exponent{'s' if k != 1 else ''} (reference)"
    return _absence_md(card)


def _kaplan_yorke_md(card: dict[str, Any]) -> str:
    if card.get("status") in ("measured", "reference", "ok"):
        return f"$D_{{KY}} = {card['value']:.4g}$"
    return _absence_md(card)


def _divergence_md(card: dict[str, Any]) -> str:
    status = card.get("status")
    if status == "ok":
        latex = card["value"]
        tag = "constant in the state" if card.get("constant") else "state-dependent"
        return f'$\\nabla\\!\\cdot f = {latex}$<br><span class="ts-prop-note">{tag}</span>'
    if status == "collapsed":
        latex = card["value"]
        note = html.escape(card.get("note", ""))
        expression = (
            f"$\\nabla\\!\\cdot f = {latex}$"
            if card.get("constant")
            else f"${latex}$ (constant part of $\\nabla\\!\\cdot f$)"
        )
        return f'{expression}<br><span class="ts-prop-note">{note}</span>'
    return _absence_md(card)


def _equilibria_md(card: dict[str, Any]) -> str:
    if card.get("status") in ("measured", "ok"):
        n = card["value"]
        kind = card.get("kind", "equilibria found")
        if n == 0:
            return "none found in this search"
        stable = card.get("stable", 0)
        unstable = card.get("unstable", 0)
        marginal = card.get("marginal", n - stable - unstable)
        return (
            f'{n} {kind}<br><span class="ts-prop-note">'
            f"{stable} stable · {unstable} unstable · {marginal} marginal / undetermined</span>"
        )
    return _absence_md(card)


def _absence_md(card: dict[str, Any]) -> str:
    status = card.get("status")
    label = "Not applicable" if status in ("not_applicable", "na") else "Not computed"
    reason = html.escape(card.get("reason", "No numerical result is available."))
    return f'<span class="ts-prop-note">{label}: {reason}</span>'


def _details_md(card: dict[str, Any], reference: dict[str, Any] | None = None) -> str:
    """Expose settings and reference conditions without crowding the answer."""
    rows = []
    if card.get("method"):
        rows.append(f"<p>{html.escape(card['method'])}</p>")
    if card.get("note"):
        rows.append(f"<p>{html.escape(card['note'])}</p>")
    if card.get("stability_test"):
        rows.append(f"<p>{html.escape(card['stability_test'])}</p>")
    for key in ("settings", "preparation", "evidence", "full_spectrum_attempt", "leading_attempt"):
        if card.get(key):
            data = html.escape(json.dumps(card[key], indent=2, ensure_ascii=False))
            rows.append(f"<p>{key.capitalize()}</p><pre><code>{data}</code></pre>")
    if "elapsed_seconds" in card:
        rows.append(
            f"<p>Estimator runtime in the recorded environment: {card['elapsed_seconds']:g} s.</p>"
        )
    if reference:
        data = html.escape(json.dumps(reference, indent=2, ensure_ascii=False))
        rows.append(
            "<p>Reference target and the library's check configuration, separate from "
            "the measurement above. The source identifies the target; "
            "<code>kwargs</code>, <code>ic</code> and <code>atol</code> describe "
            "the local validation run. They are not published confidence intervals "
            "or a claim that the source used those solver settings.</p>"
            f"<pre><code>{data}</code></pre>"
        )
    if not rows:
        return ""
    return (
        '<details class="ts-prop-detail"><summary>Method and evidence</summary>'
        + "".join(rows)
        + "</details>"
    )


def to_markdown(props: dict[str, Any]) -> str:
    """Render four readable cards with their status and expandable evidence."""
    cards = [
        ("Lyapunov spectrum", "lyapunov_spectrum", _lyapunov_md),
        ("Kaplan–Yorke dimension", "kaplan_yorke", _kaplan_yorke_md),
        (
            "Instantaneous Jacobian trace"
            if props.get("family") == "dde"
            else "Drift Jacobian trace"
            if props.get("family") == "sde"
            else "Divergence ∇·f",
            "divergence",
            _divergence_md,
        ),
        ("Equilibria / fixed points", "equilibria", _equilibria_md),
    ]
    labels = {
        "measured": "Finite-time / numerical estimate",
        "reference": "Reference",
        "ok": "Symbolic",
        "collapsed": "Symbolic summary",
    }
    lines = ['<div class="ts-props" markdown>']
    for label, key, render in cards:
        card = props[key]
        status = labels.get(card.get("status"), "")
        lines += [
            '<div class="ts-prop" markdown>',
            f'<div class="ts-prop-label">{label}</div>',
            f'<div class="ts-prop-status">{status}</div>' if status else "",
            f'<div class="ts-prop-value">{render(card)}</div>',
            _details_md(card, props.get("reference") if key == "lyapunov_spectrum" else None),
            "</div>",
        ]
    lines.append("</div>")
    if props.get("provenance"):
        data = html.escape(json.dumps(props["provenance"], indent=2, ensure_ascii=False))
        lines.append(
            '<details class="ts-prop-detail"><summary>Recorded parameters and software</summary>'
            f"<pre><code>{data}</code></pre></details>"
        )
    return "\n".join(lines)
