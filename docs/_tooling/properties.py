"""
Build-time **computed property cards** for the per-system documentation pages.

Each system page carries a "Properties" panel with four stat cards:

1. **Lyapunov spectrum** — the maximal/full set of Lyapunov exponents.  The
   **literature** value (the ``known_lyapunov`` ClassVar) is preferred when
   present; otherwise it is computed via the system's own
   ``lyapunov_spectrum`` with a bounded budget.  An ``n_positive``-only
   ``known_lyapunov`` (no published spectrum) is reported as "≥ k positive
   exponents" — never a fabricated number.
2. **Kaplan–Yorke dimension** — derived from the spectrum
   (``ts.kaplan_yorke_dimension``); ``todo`` when the spectrum is unknown.
3. **Phase-space divergence ∇·f** — the *symbolic* trace of the Jacobian
   (``Σ ∂f_i/∂y_i``) rendered as LaTeX, straight from the system's symbolic
   ``_equations`` / ``_drift``.  Constant ⇒ uniform contraction/expansion;
   state-dependent ⇒ shown with its free symbols.  For **maps** the flow
   divergence is undefined, so this card reports the discrete analogue
   *qualitatively* (it is skipped with a clear note rather than guessed).
4. **Equilibria** — a best-effort count + stability summary from
   ``ts.fixed_points``; ``todo`` when the root finder does not converge or the
   system is variable-dimensional.

Never fabricate a number
------------------------
Every quantity that is ill-defined, fails to converge, or would be too slow to
compute at build time degrades to an explicit ``{"status": "todo", "reason":
...}`` sentinel.  :func:`to_markdown` renders those as a clearly-marked "TODO"
card, so a reader never mistakes a placeholder for a measurement.

Caching
-------
Results are cached on disk under ``.cache/docs-props`` keyed by
``sha256(class source ‖ default params ‖ this module's version)``, so an
unchanged system is a cheap JSON read (no re-integration).  CI persists the
directory between builds (mirroring :mod:`figures` / :mod:`threejs_viewer`).

Environment flags
-----------------
- ``TSD_DOCS_PROPS=0`` — skip *all* expensive recompute: every computed
  quantity (Lyapunov / Kaplan–Yorke / equilibria) returns its ``todo``
  sentinel, while the cheap symbolic divergence still renders.  A fast,
  number-free preview.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache" / "docs-props"

#: Bump when the computed-quantity logic or card shaping materially changes
#: (cache buster — invalidates every on-disk props payload).
PROPS_VERSION = "1"

#: Truthy ``TSD_DOCS_PROPS`` (default) computes the cards; ``"0"`` makes every
#: *expensive* quantity a TODO (number-free preview).  The symbolic divergence
#: is cheap and always rendered.
_COMPUTE = os.environ.get("TSD_DOCS_PROPS", "1") != "0"

#: Bounded Lyapunov budget for the **computed** path (when there is no
#: literature ``known_lyapunov``).  Kept modest so a cold full-catalogue build
#: stays in the minutes, not hours; a system with a published spectrum never
#: pays this.
_LYAP_BUDGET: dict[str, Any] = {"final_time": 120.0, "dt": 0.1, "burn_in": 40.0}

#: Hard wall-clock kill for the computed-Lyapunov **child process**.  A few
#: off-attractor catalogue systems spiral inside the Rust variational integrator
#: (uninterruptible from Python); process isolation + this kill keep the docs
#: build from ever stalling.  A killed child → a clear TODO card.
_LYAP_TIMEOUT_S = 20.0

#: Above this fixed dimension the computed Lyapunov path is declared too slow
#: (a full-spectrum QR over many tangent vectors); such systems fall back to a
#: literature value if present, else a TODO.  Variable-dim (``dim is None``)
#: systems are always TODO for the computed path.
_LYAP_MAX_DIM = 6

#: Equilibrium search is skipped above this dimension (multi-start Newton over a
#: high-dim state is slow and rarely converges cleanly in a docs budget).
_FP_MAX_DIM = 8


# --------------------------------------------------------------------------- #
# Cache plumbing
# --------------------------------------------------------------------------- #
def cache_key(entry) -> str:
    """Content hash: class source + default params + module version + compute flag.

    The compute flag is part of the key so a ``TSD_DOCS_PROPS=0`` (TODO-only)
    payload never masquerades as a fully-computed one in the cache.
    """
    cls_src = inspect.getsource(entry.cls)
    params = repr(sorted((entry.params or {}).items()))
    flag = "compute" if _COMPUTE else "todo"
    blob = cls_src + params + PROPS_VERSION + flag
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _cache_path(entry) -> pathlib.Path:
    return CACHE_DIR / f"{entry.name}-{cache_key(entry)}.json"


# --------------------------------------------------------------------------- #
# Sentinels
# --------------------------------------------------------------------------- #
def _todo(reason: str) -> dict[str, Any]:
    """Build a clearly-marked placeholder — never confused with a measured value."""
    return {"status": "todo", "reason": reason}


def _ok(value: Any, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "ok", "value": value}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# Lyapunov spectrum + Kaplan–Yorke
# --------------------------------------------------------------------------- #
def _known_lyapunov(entry) -> dict[str, Any] | None:
    """Read the literature ``known_lyapunov`` ClassVar into a card, if present.

    Two shapes are supported (mirroring the registry metadata):

    - ``{"spectrum": (...), ...}`` — a published numeric spectrum (preferred):
      reported verbatim with its ``source``.
    - ``{"n_positive": k, ...}`` — only the *count* of positive exponents is
      known: reported as "≥ k positive" with **no fabricated numbers**.
    """
    kl = getattr(entry.cls, "known_lyapunov", None)
    if not kl:
        return None
    source = kl.get("source")
    if "spectrum" in kl:
        spectrum = [float(v) for v in kl["spectrum"]]
        return _ok(spectrum, origin="literature", source=source)
    if "n_positive" in kl:
        return {
            "status": "partial",
            "n_positive": int(kl["n_positive"]),
            "origin": "literature",
            "source": source,
            "reason": "only the number of positive exponents is published",
        }
    return None


def _compute_lyapunov(entry) -> dict[str, Any]:
    """Compute a full Lyapunov spectrum on a bounded budget (no literature value).

    DDE/SDE and variable-/high-dimensional systems are declared out of the
    docs-build budget and return a TODO rather than a slow or ill-posed run.

    The integration runs in a **child process** with a hard wall-clock timeout
    (:data:`_LYAP_TIMEOUT_S`).  A handful of off-attractor catalogue systems
    send the variational integration into a slow diverging spiral *inside the
    Rust engine*, which a Python ``signal`` alarm cannot interrupt — so the only
    robust guard against a stalled docs build is process isolation with a kill.
    A timeout / crash / non-finite result all degrade to a clear TODO.
    """
    if not _COMPUTE:
        return _todo("expensive recompute disabled (TSD_DOCS_PROPS=0)")
    if entry.family in ("dde", "sde"):
        # DDE Lyapunov is the infinite-dim-history estimator (its own budget);
        # SDE exponents are noise-driven — neither is a stable build-time card.
        return _todo(f"{entry.family.upper()} Lyapunov not computed at build time")
    if entry.dim is None:
        return _todo("variable-dimensional system — spectrum not computed")
    if entry.dim > _LYAP_MAX_DIM:
        return _todo(f"dim {entry.dim} > {_LYAP_MAX_DIM} — full spectrum too slow")
    return _spectrum_in_subprocess(entry.name)


def _spectrum_in_subprocess(name: str) -> dict[str, Any]:
    """Run ``_lyapunov_worker(name)`` in a child with a hard wall-clock kill.

    Returns the ``ok`` / ``todo`` card the worker printed as JSON, or a timeout
    TODO if the child overran (an uninterruptible Rust spiral) — the child is
    killed, so the parent build is never blocked.
    """
    import subprocess
    import sys

    cmd = [sys.executable, "-c", _WORKER_SRC, name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=_LYAP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return _todo(f"Lyapunov compute exceeded {_LYAP_TIMEOUT_S:.0f}s (killed)")
    except Exception as exc:  # noqa: BLE001
        return _todo(f"Lyapunov subprocess failed: {type(exc).__name__}")
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return _todo("Lyapunov compute produced no result")
    try:
        return json.loads(line[-1])
    except Exception:  # noqa: BLE001
        return _todo("Lyapunov compute returned an unparseable result")


#: Source for the isolated Lyapunov worker (run as ``python -c``).  Kept as a
#: string so the child is a clean interpreter with no inherited import state; it
#: prints exactly one JSON card on stdout.
_WORKER_SRC = """
import json, sys
sys.path.insert(0, "docs/_tooling")
import properties as P
from tsdynamics import registry

name = sys.argv[1]
entry = next(e for e in registry.all_systems() if e.name == name)
print(json.dumps(P._lyapunov_worker(entry)))
"""


def _lyapunov_worker(entry) -> dict[str, Any]:
    """Run the actual (potentially slow) Lyapunov integration — in a child only.

    Separated so the timeout wrapper drives it via ``python -c``; calling it in
    the parent risks the uninterruptible Rust spiral the subprocess guards
    against.
    """
    try:
        sys_obj = entry.cls()
        kwargs = dict(_LYAP_BUDGET) if entry.family == "ode" else {}
        spectrum = sys_obj.lyapunov_spectrum(**kwargs)
        vals = [float(v) for v in list(spectrum)]
    except Exception as exc:  # noqa: BLE001 — soft-fail to a TODO card
        return _todo(f"lyapunov_spectrum failed: {type(exc).__name__}")
    if not vals or any(not _finite(v) for v in vals):
        return _todo("non-finite spectrum (divergent / off-attractor)")
    return _ok(vals, origin="computed")


def _finite(v: float) -> bool:
    return v == v and abs(v) != float("inf")


def _lyapunov_card(entry) -> dict[str, Any]:
    """Lyapunov card: literature value preferred, else a bounded computation."""
    known = _known_lyapunov(entry)
    if known is not None and known.get("status") in ("ok", "partial"):
        # A literature spectrum is authoritative; an ``n_positive``-only entry is
        # "partial" — still better than a from-scratch compute we'd then have to
        # truncate.  Either way, prefer the published metadata.
        return known
    return _compute_lyapunov(entry)


def _kaplan_yorke_card(lyap: dict[str, Any]) -> dict[str, Any]:
    """Kaplan–Yorke dimension from a spectrum card (``todo`` if no spectrum)."""
    if lyap.get("status") != "ok":
        return _todo("requires a numeric Lyapunov spectrum")
    spectrum = lyap["value"]
    try:
        import tsdynamics as ts

        ky = float(ts.kaplan_yorke_dimension(spectrum))
    except Exception as exc:  # noqa: BLE001
        return _todo(f"kaplan_yorke_dimension failed: {type(exc).__name__}")
    if not _finite(ky):
        return _todo("Kaplan–Yorke undefined for this spectrum")
    return _ok(ky, origin=lyap.get("origin", "computed"))


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

    Built from named state symbols (so the LaTeX matches the equations card).
    For a DDE the delayed terms ``y(i, t-τ)`` are independent function symbols,
    so differentiating w.r.t. the instantaneous state gives the *instantaneous*
    divergence — the right local contraction rate.  Raises on any system whose
    RHS we cannot lower symbolically (variable-dim, NumPy bodies, …); the caller
    turns that into a TODO.
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
    return div


def _divergence_card(entry) -> dict[str, Any]:
    """Divergence card: LaTeX of ∇·f for a flow; a qualitative note for a map.

    The symbolic trace is cheap, so it is computed regardless of
    ``TSD_DOCS_PROPS`` (it is not an integration).  A map has no continuous-flow
    divergence — the discrete analogue is the Jacobian *determinant* (the
    per-step phase-space volume factor), which we deliberately do not render as
    ``∇·f`` and instead mark as not-applicable with a pointer.
    """
    if entry.family == "map":
        return {
            "status": "na",
            "reason": "discrete map — flow divergence undefined (per-step contraction is |det J|)",
        }
    try:
        div = _divergence_expr(entry)
    except Exception as exc:  # noqa: BLE001 — symbolic lowering failed
        return _todo(f"symbolic divergence unavailable: {type(exc).__name__}")
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
        constant = len(div.free_symbols) == 0 or _is_constant_in_state(entry, div)
    except Exception as exc:  # noqa: BLE001
        return _todo(f"divergence render failed: {type(exc).__name__}")
    return _ok(latex, constant=constant)


def _has_nonsmooth_derivative(div) -> bool:
    """Whether ``div`` carries an unresolved non-smooth derivative artifact.

    A ``Derivative(abs(...))`` / ``sign'`` / ``Subs(...)`` left in the symbolic
    trace means the RHS is non-smooth (``abs``/``sign``/fractional power); such a
    divergence is piecewise and not a clean closed-form card.
    """
    text = str(div)
    return "Derivative" in text or "Subs" in text or "sign(" in text.lower()


def _is_constant_in_state(entry, div) -> bool:
    """Whether ∇·f is free of any *state* symbol (i.e. parameter-only constant)."""
    try:
        import symengine

        sys_obj = entry.cls()
        dim = sys_obj.dim or 0
        names = _state_names(entry.cls, dim)
        state_syms = {symengine.Symbol(n) for n in names}
        return not (set(div.free_symbols) & state_syms)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Equilibria (best-effort)
# --------------------------------------------------------------------------- #
def _equilibria_card(entry) -> dict[str, Any]:
    """Best-effort fixed-point count + stability split via ``ts.fixed_points``.

    A non-convergent search, a too-large state, or an SDE all degrade to a TODO
    — the count is never guessed.
    """
    if not _COMPUTE:
        return _todo("expensive recompute disabled (TSD_DOCS_PROPS=0)")
    if entry.family == "sde":
        return _todo("equilibria of a stochastic system not enumerated")
    if entry.dim is None:
        return _todo("variable-dimensional system — equilibria not enumerated")
    if entry.dim > _FP_MAX_DIM:
        return _todo(f"dim {entry.dim} > {_FP_MAX_DIM} — equilibrium search skipped")
    try:
        import tsdynamics as ts

        sys_obj = entry.cls()
        fps = ts.fixed_points(sys_obj)
    except Exception as exc:  # noqa: BLE001 — root finder did not converge
        return _todo(f"fixed_points failed: {type(exc).__name__}")
    items = list(fps)
    n = len(items)
    stable = sum(1 for fp in items if bool(getattr(fp, "stable", False)))
    kind = "fixed points" if entry.family == "map" else "equilibria"
    return _ok(n, stable=stable, unstable=n - stable, kind=kind, origin="computed")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_properties(entry) -> dict[str, Any]:
    """Compute (or load from cache) the four property cards for ``entry``.

    Returns a dict with keys ``lyapunov_spectrum``, ``kaplan_yorke``,
    ``divergence``, ``equilibria`` — each a status-tagged card
    (``ok`` / ``partial`` / ``na`` / ``todo``).  Fast on a cache hit (a JSON
    read); a miss computes the cards and writes the cache.  ``TSD_DOCS_PROPS=0``
    short-circuits every expensive quantity to a TODO while still rendering the
    cheap symbolic divergence.
    """
    cached = _load_cache(entry)
    if cached is not None:
        return cached

    lyap = _lyapunov_card(entry)
    props = {
        "name": entry.name,
        "family": entry.family,
        "lyapunov_spectrum": lyap,
        "kaplan_yorke": _kaplan_yorke_card(lyap),
        "divergence": _divergence_card(entry),
        "equilibria": _equilibria_card(entry),
    }
    _store_cache(entry, props)
    return props


def _load_cache(entry) -> dict[str, Any] | None:
    path = _cache_path(entry)
    if not path.exists():
        return None
    try:
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
    if status == "ok":
        spectrum = card["value"]
        body = "$" + ",\\; ".join(_fmt_exponent(v) for v in spectrum) + "$"
        src = card.get("source")
        if card.get("origin") == "literature" and src:
            body += f'<br><span class="ts-prop-src">{src}</span>'
        elif card.get("origin") == "computed":
            body += '<br><span class="ts-prop-src">computed at build</span>'
        return body
    if status == "partial":
        k = card.get("n_positive", "?")
        src = card.get("source", "")
        note = f" — {src}" if src else ""
        return (
            f"≥ {k} positive exponent{'s' if k != 1 else ''} "
            f'<span class="ts-prop-src">(literature{note})</span>'
        )
    return _todo_md(card)


def _kaplan_yorke_md(card: dict[str, Any]) -> str:
    if card.get("status") == "ok":
        return f"$D_{{KY}} = {card['value']:.4g}$"
    return _todo_md(card)


def _divergence_md(card: dict[str, Any]) -> str:
    status = card.get("status")
    if status == "ok":
        latex = card["value"]
        tag = "constant" if card.get("constant") else "state-dependent"
        return f'$\\nabla\\!\\cdot f = {latex}$<br><span class="ts-prop-src">{tag}</span>'
    if status == "na":
        return f'<span class="ts-prop-na">n/a — {card.get("reason", "")}</span>'
    return _todo_md(card)


def _equilibria_md(card: dict[str, Any]) -> str:
    if card.get("status") == "ok":
        n = card["value"]
        kind = card.get("kind", "equilibria")
        if n == 0:
            return f'none found <span class="ts-prop-src">(no {kind})</span>'
        stable = card.get("stable", 0)
        unstable = card.get("unstable", n - stable)
        return (
            f'{n} {kind}<br><span class="ts-prop-src">{stable} stable · {unstable} unstable</span>'
        )
    return _todo_md(card)


def _todo_md(card: dict[str, Any]) -> str:
    reason = card.get("reason", "not available")
    return f'<span class="ts-prop-todo">TODO — {reason}</span>'


def to_markdown(props: dict[str, Any]) -> str:
    """Render the four computed property cards as a Markdown/HTML block.

    Emits a ``.ts-props`` grid of four ``.ts-prop`` cards (Lyapunov spectrum,
    Kaplan–Yorke dimension, phase-space divergence, equilibria).  Math is KaTeX
    (``$…$`` arithmatex); TODO / n-a quantities render as clearly-marked muted
    spans so a placeholder is never mistaken for a measurement.
    """
    cards = [
        ("Lyapunov spectrum", _lyapunov_md(props["lyapunov_spectrum"])),
        ("Kaplan–Yorke dimension", _kaplan_yorke_md(props["kaplan_yorke"])),
        ("Divergence ∇·f", _divergence_md(props["divergence"])),
        ("Equilibria", _equilibria_md(props["equilibria"])),
    ]
    lines = ['<div class="ts-props" markdown>']
    for label, value in cards:
        lines += [
            '<div class="ts-prop" markdown>',
            f'<div class="ts-prop-label">{label}</div>',
            f'<div class="ts-prop-value">{value}</div>',
            "</div>",
        ]
    lines.append("</div>")
    return "\n".join(lines)
