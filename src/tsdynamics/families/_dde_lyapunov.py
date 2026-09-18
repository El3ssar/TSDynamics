"""Engine-based Lyapunov spectrum for delay systems (stream E-DDE-LYAP).

The Lyapunov spectrum of a DDE is the time-averaged logarithmic stretch of a set
of deviation *functions* carried along a trajectory.  Unlike an ODE — whose
tangent space is the finite ``R^dim`` — a DDE's tangent space is the
**infinite-dimensional history space** ``C([-τ, 0], R^dim)``, so a deviation is a
function on the delay window and there can be more exponents than state
components (``k`` may exceed ``dim``).

The v2 path computes this with JiTCDDE (``jitcdde_lyap``), removed at the M3
migration gate.  This module is the engine-based replacement, mirroring the ODE
variational core (:mod:`tsdynamics.derived._variational`): it builds the
**extended** DDE — base state ⊕ ``k`` deviation states, the deviation equations
being the symbolic variational dynamics

    δy'(t) = Σ_j (∂f/∂y_j) · δy_j(t)  +  Σ_s (∂f/∂y_{c(s)}(t-τ_s)) · δy_{c(s)}(t-τ_s)

(a per-current-state Jacobian plus one Jacobian per delay slot) — and integrates
it on the existing Rust DDE engine (:func:`tsdynamics.engine.run.integrate`,
``backend="interp"/"jit"``).  The variational equations are ordinary delayed RHS
expressions, so the **frozen IR is untouched** — the delayed deviations are just
extra delay slots, exactly like the base system's delays.

Benettin renormalisation in the function space
-----------------------------------------------
Because the deviation is a function, the QR is taken over its **history segment**
``[t-τ_max, t]`` (sampled on the output grid), not over the current value — that
is what lets ``k`` exceed ``dim``.  The engine integrates one chunk of length
``T = τ_max`` (the minimal length that yields a full past segment), the ``k``
deviation segments are QR-orthonormalised, ``log|diag R|`` is accumulated, and
the orthonormalised segments seed the next chunk's history (a cubic-spline past).
With ``T = τ_max`` and ``dt`` dividing ``τ_max`` the segment sample nodes coincide
with the engine output grid, so the reseed reproduces the stored nodes exactly (no
resampling-at-nodes error); the scheme is otherwise uniformly O(dt⁴) (the
cubic-spline past and the engine's Hermite dense output interpolate between nodes).
The deviation recombination is exact at the function level — the variational
dynamics is linear, so a linear combination of deviation solutions is a solution —
acting on that sampled representative.  When ``dt`` does not divide ``τ_max`` the
reseed falls back to off-grid interpolation (a warning is emitted).

Validated (reference-free) on the five built-in DDEs (and a 2-D synthetic DDE) —
the Mackey–Glass leading exponent is positive, matching its ``known_lyapunov``
``n_positive=1``; ``interp`` and ``jit`` agree bit-for-bit. (The original
Rust-vs-``jitcdde_lyap`` parity gate ran before JiTCDDE was removed.)
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from tsdynamics.errors import ConvergenceError, InvalidParameterError, invalid_value
from tsdynamics.utils.tolerances import DDE_LYAPUNOV_ATOL, DDE_LYAPUNOV_RTOL

__all__ = ["dde_lyapunov_spectrum"]

#: Ceiling on the extended system's dimension, ``dim * (1 + k)``.
#:
#: :func:`_build_extended_tape` loops *symbolically* over ``k``, emitting
#: ``dim`` variational expressions per deviation, so its cost grows with the
#: extended dimension and it burns that cost in SymEngine long before any Rust
#: code runs.  An absurd request (``k=2**34``) therefore did not fail — it
#: sat in the symbolic builder indefinitely, which reads to a user as a hang.
#:
#: 10 000 is deliberately far above any defensible request and still lowers in
#: seconds: a DDE's tangent space is infinite-dimensional, but the exponents are
#: only resolvable up to the delay-window sampling ``(n_seg+1)·dim`` (checked
#: separately below), and nobody interprets the ten-thousandth exponent of a
#: delay system.  It exists to turn a hang into a message, not to police
#: modelling choices.
_MAX_EXTENDED_DIM = 10_000


def _build_extended_tape(system: Any, k: int) -> tuple[Any, list[Any], int]:
    """Lower the extended variational DDE of ``system`` with ``k`` deviations.

    Returns ``(tape, delay_slots, dim)``.  The tape is an ordinary RHS over
    ``dim·(k+1)`` state inputs plus the delay slots: the base system's slots
    (reused by the deviation equations) followed by one delayed-deviation slot
    per ``(deviation, base-slot)`` pair.  Parameters are folded to constants,
    matching :func:`tsdynamics.engine.compile.lower_dde`.
    """
    from collections.abc import Callable

    import symengine

    from tsdynamics.engine.compile import (
        DelaySlot,
        _is_past_y,
        _past_y_component_and_delay,
        lower_expressions,
    )
    from tsdynamics.engine.symbols import state_time_symbols
    from tsdynamics.families.continuous import (
        _resolve_derivative_nodes as _resolve_derivative_nodes_untyped,
    )

    # ``_resolve_derivative_nodes`` lives in another module without a public type
    # annotation; bind it through a typed local so the calls below are typed.
    resolve_derivative_nodes: Callable[[Any], Any] = _resolve_derivative_nodes_untyped

    y, t_sym = state_time_symbols()

    dim = system.dim
    exprs = list(type(system)._equations(y, t_sym, **system.params.as_dict()))
    if len(exprs) != dim:
        raise InvalidParameterError(f"_equations must return {dim} expressions, got {len(exprs)}")

    t_canon = symengine.Symbol("t")
    u = [symengine.Symbol(f"u{i}") for i in range(dim)]

    # First pass: collect the base (component, delay) slots and a node→slot map.
    base_slots: list[tuple[int, float]] = []
    slot_index: dict[tuple[int, float], int] = {}
    node_to_key: dict[Any, tuple[int, float]] = {}
    current_subs: dict[Any, Any] = {}  # explicit ``y(i, t)`` current-time accesses

    def scan(node: Any) -> None:
        node = symengine.sympify(node)
        if _is_past_y(node):
            comp, delay = _past_y_component_and_delay(node, t_sym, system)
            if delay == 0.0:
                current_subs[node] = u[comp]  # ``y(i, t)`` is the current state
                return
            key = (comp, float(delay))
            if key not in slot_index:
                slot_index[key] = len(base_slots)
                base_slots.append(key)
            node_to_key[node] = key
            return
        for arg in node.args:
            scan(arg)

    for e in exprs:
        scan(e)

    nb = len(base_slots)
    if nb == 0:
        raise InvalidParameterError(f"{type(system).__name__}: no delayed terms — not a DDE")

    # Symbols for the base delayed terms, then the base RHS in (u, delayed) form.
    bdel = [symengine.Symbol(f"ud{s}") for s in range(nb)]
    subs = {y(i): u[i] for i in range(dim)}
    subs[t_sym] = t_canon
    subs.update(current_subs)  # explicit y(i, t) → current state
    for node, key in node_to_key.items():
        subs[node] = bdel[slot_index[key]]
    f = [symengine.sympify(e).subs(subs) for e in exprs]

    # Jacobians (a.e.-resolved abs/sign): wrt current state and wrt each delay slot.
    df_du = [
        [resolve_derivative_nodes(symengine.sympify(f[i]).diff(u[j])) for j in range(dim)]
        for i in range(dim)
    ]
    df_dd = [
        [resolve_derivative_nodes(symengine.sympify(f[i]).diff(bdel[s])) for s in range(nb)]
        for i in range(dim)
    ]

    # Deviation current + delayed symbols, and the variational equations.
    dev = [[symengine.Symbol(f"w{m}_{j}") for j in range(dim)] for m in range(k)]
    devd = [[symengine.Symbol(f"wd{m}_{s}") for s in range(nb)] for m in range(k)]
    exprs_ext: list[Any] = list(f)
    for m in range(k):
        for i in range(dim):
            acc = symengine.Integer(0)
            for j in range(dim):
                acc = acc + df_du[i][j] * dev[m][j]
            for s in range(nb):
                acc = acc + df_dd[i][s] * devd[m][s]
            exprs_ext.append(acc)

    # Input layout: extended state (base ⊕ k deviations), then delay slots
    # (base slots, then each deviation's delayed slots).
    state_syms: list[Any] = list(u)
    for m in range(k):
        state_syms.extend(dev[m])
    n_state = len(state_syms)

    slot_syms: list[Any] = list(bdel)
    slots: list[Any] = [
        DelaySlot(input_index=n_state + s, component=base_slots[s][0], delay=base_slots[s][1])
        for s in range(nb)
    ]
    pos = nb
    for m in range(k):
        for s in range(nb):
            comp, delay = base_slots[s]
            slot_syms.append(devd[m][s])
            slots.append(
                DelaySlot(input_index=n_state + pos, component=dim + m * dim + comp, delay=delay)
            )
            pos += 1

    tape = lower_expressions(exprs_ext, [*state_syms, *slot_syms], time_sym=t_canon)
    return tape, slots, dim


def _seed_deviations(n_seg: int, k: int, dim: int, grid: np.ndarray) -> np.ndarray:
    """Build ``k`` distinct, QR-orthonormal initial deviation history segments.

    Distinct cosines over the delay window give linearly-independent seed
    functions even when ``k > dim`` (the infinite-dimensional case); the
    burn-in then rotates them onto the leading covariant Lyapunov directions.
    """
    span = grid[-1] - grid[0]
    dev = np.zeros((n_seg + 1, k, dim))
    for m in range(k):
        wave = np.cos((m + 1) * np.pi * (grid - grid[0]) / (span + 1e-300))
        # Seed deviation m on state component m % dim with a distinct waveform, so
        # the k seed functions are linearly independent for any (k, dim).
        dev[:, m, m % dim] = wave
    return _qr_segments(dev, k, dim)[0]


def _qr_segments(dev: np.ndarray, k: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """QR-orthonormalise ``k`` deviation history segments over the function space.

    ``dev`` has shape ``(n_seg+1, k, dim)`` (point, deviation, component); each
    deviation's flattened ``(point, component)`` history is one column of an
    ``((n_seg+1)·dim, k)`` matrix (its function-space representative), which is
    QR-decomposed.  The deviation axis must be the *column* axis and the component
    axis kept adjacent to the point axis (``transpose(0, 2, 1)``) — a plain
    ``reshape`` would interleave the deviation and component axes and scramble the
    columns for ``dim > 1``.  Returns the orthonormal segments (same shape) and
    ``log|diag R|`` (the per-deviation log growth over the chunk).
    """
    n_pts = dev.shape[0]
    mat = dev.transpose(0, 2, 1).reshape(n_pts * dim, k)
    q, r = np.linalg.qr(mat)
    # Floor the per-deviation growth at the smallest positive normal float before
    # the log: a rank-deficient or collapsed deviation gives a (near-)zero ``R``
    # diagonal, and ``log(0) = -inf`` both poisons the time-average and raises a
    # ``divide by zero`` RuntimeWarning that is *fatal* under
    # ``filterwarnings=error`` (the test profile).  The floor is a guard only — a
    # healthy renormalisation has ``|R_ii| >> tiny``, so it never perturbs a
    # converged spectrum (``interp == jit`` stays bit-for-bit).
    diag = np.maximum(np.abs(np.diag(r)), np.finfo(np.float64).tiny)
    log_growth = np.log(diag)
    return q.reshape(n_pts, dim, k).transpose(0, 2, 1), log_growth


def dde_lyapunov_spectrum(
    system: Any,
    *,
    k: int = 1,
    final_time: float = 200.0,
    dt: float = 0.1,
    burn_in: float = 50.0,
    ic: Any = None,
    backend: str = "jit",
    rtol: float = DDE_LYAPUNOV_RTOL,
    atol: float = DDE_LYAPUNOV_ATOL,
) -> np.ndarray:
    """Estimate the ``k`` leading Lyapunov exponents of a DDE on the engine.

    The function-space Benettin estimator described in the module docstring:
    integrate the extended variational DDE on the Rust engine in chunks of one
    delay window, QR-renormalise the deviation history segments each chunk, and
    time-average ``log|diag R|`` after burn-in.

    Parameters
    ----------
    system : DelaySystem
        The delay system.
    k : int, default 1
        Number of leading exponents (may exceed ``dim`` — the tangent space is
        infinite-dimensional).
    final_time, dt, burn_in : float
        Post-burn-in averaging window, output/renormalisation step, and discarded
        transient.  ``final_time`` is the window over which ``log|diag R|`` is
        time-averaged *after* ``burn_in`` — the total integration is
        ``burn_in + final_time`` — so growing ``burn_in`` discards more transient
        without shrinking the average.  ``dt`` should divide the maximum delay for
        an exact base-history reuse.
    ic : array-like, optional
        Base initial state; pass the end state of a prior ``integrate`` so the
        run starts on the attractor (strongly recommended).
    backend : {"jit", "interp"}, default "jit"
        The compiled engine's Cranelift JIT (default, served from the
        compiled-evaluator cache) or its bit-for-bit identical SSA-tape
        interpreter.  ``"reference"`` is rejected (the engine has no pure-Python
        DDE integrator), matching DDE integration.

        .. versionchanged:: 6.0
            Default moved from ``"interp"`` to ``"jit"``.
    rtol, atol : float
        Engine integration tolerances.  Default
        :data:`~tsdynamics.utils.tolerances.DDE_LYAPUNOV_RTOL` /
        :data:`~tsdynamics.utils.tolerances.DDE_LYAPUNOV_ATOL`.

    Returns
    -------
    ndarray, shape (k,)
        Exponents in descending order.
    """
    from scipy.interpolate import CubicSpline

    from tsdynamics.engine.problem import DDEProblem
    from tsdynamics.engine.run import integrate, resolve_backend

    # Bound `k` BEFORE `_build_extended_tape`, which is where an absurd value
    # is spent: it loops symbolically over `k` building the extended DDE, so
    # `k=2**34` never reached Rust — it disappeared into SymEngine and looked
    # like a hang.  All three checks are up-front and name the offending value.
    try:
        k = operator.index(k)
    except TypeError:
        # `int(k)` used to silently truncate here, so `k=2.7` quietly
        # computed 2 exponents. A non-integer count is a mistake, not a rounding.
        raise invalid_value(
            "k",
            k,
            rule="must be an integer (a count of exponents)",
        ) from None
    if k < 1:
        raise invalid_value("k", k, rule="must be >= 1 (a count of exponents)")
    n_ext_requested = int(system.dim) * (1 + k)
    if n_ext_requested > _MAX_EXTENDED_DIM:
        raise invalid_value(
            "k",
            k,
            rule=(
                f"would build an extended system of dimension dim*(1+k)="
                f"{n_ext_requested}, over the {_MAX_EXTENDED_DIM} ceiling"
            ),
            hint=(
                "The extended DDE carries the base state plus one deviation state "
                "per exponent and is built symbolically, so this would take "
                "effectively forever. Request fewer exponents."
            ),
        )
    backend = resolve_backend(backend)
    if backend == "reference":
        raise NotImplementedError(
            "DDE Lyapunov has no pure-Python reference integrator; use "
            "backend='interp'/'jit' (the Rust engine)."
        )

    tape, slots, dim = _build_extended_tape(system, k)
    max_delay = max(s.delay for s in slots)
    n_ext = dim * (k + 1)

    # Chunk length = one delay window (minimal length giving a full past segment),
    # so the past window coincides with the engine output grid.
    chunk = max_delay
    n_seg = max(1, int(round(max_delay / dt)))
    grid = np.linspace(-max_delay, 0.0, n_seg + 1)

    # The deviation history segment has (n_seg+1)·dim sample dimensions, the ceiling
    # on the number of linearly-independent deviation functions it can carry.
    if k > (n_seg + 1) * dim:
        raise InvalidParameterError(
            f"k={k} exceeds the delay-window resolution (n_seg+1)·dim="
            f"{(n_seg + 1) * dim}; decrease dt (finer window) or lower k."
        )
    # The base-history reuse is exact only when dt divides the maximum delay (then
    # the segment grid coincides with the engine output grid); otherwise the reseed
    # falls back to O(dt²) interpolation off-grid.
    if abs(max_delay / dt - round(max_delay / dt)) > 1e-9:
        import warnings

        warnings.warn(
            f"dt={dt} does not divide the maximum delay {max_delay}; the history "
            "reseed falls back to interpolation off the output grid (less accurate). "
            "Choose a dt that divides the delay for the exact-reuse path.",
            stacklevel=2,
        )

    base_ic = np.asarray(system.resolve_ic(ic), dtype=np.float64).ravel()
    base_seg = np.tile(base_ic, (n_seg + 1, 1))
    dev_seg = _seed_deviations(n_seg, k, dim, grid)

    # ``final_time`` is the post-burn-in AVERAGING WINDOW (the documented
    # contract), so the total integration is ``burn_in + final_time``: discard
    # ``n_burn`` chunks of transient, then average over a *full* ``final_time``
    # window of ``n_avg`` chunks.  (The previous code treated ``final_time`` as
    # the total length and carved ``burn_in`` out of it — a large ``burn_in``
    # then collapsed the average to a single delay-window FTLE, which fluctuates
    # in sign and reported a negative leading exponent for the chaotic
    # Mackey–Glass attractor.)
    n_avg = max(1, int(round(final_time / chunk)))
    # The transient is discarded a full delay window at a time, so ``n_burn`` is
    # ``round(burn_in / chunk)``.  For a large delay (``chunk = max_delay``) a
    # requested ``burn_in`` smaller than half a window rounds to zero — e.g.
    # ``tau = 100, burn_in = 50`` gives ``round(0.5) = 0`` — and the transient is
    # then NOT discarded, the exact failure the module docstring warns against.
    # Round UP to at least one window whenever a positive ``burn_in`` was asked
    # for, so a non-zero request always discards at least the first delay window.
    n_burn = int(round(burn_in / chunk))
    if burn_in > 0.0 and n_burn == 0:
        n_burn = 1
    n_chunks = n_burn + n_avg
    log_sums = np.zeros(k)
    seg_t = chunk + grid  # absolute times of the next past window within [0, chunk]

    problem = DDEProblem(tape=tape, delay_slots=slots, ic=np.zeros(n_ext), system=system)

    for c in range(n_chunks):
        spl_base = [CubicSpline(grid, base_seg[:, d]) for d in range(dim)]
        spl_dev = [[CubicSpline(grid, dev_seg[:, m, d]) for d in range(dim)] for m in range(k)]

        def history(
            s: float,
            spl_base: list[Any] = spl_base,
            spl_dev: list[list[Any]] = spl_dev,
        ) -> np.ndarray:
            out = np.empty(n_ext)
            for d in range(dim):
                out[d] = spl_base[d](s)
            for m in range(k):
                for d in range(dim):
                    out[dim + m * dim + d] = spl_dev[m][d](s)
            return out

        traj = integrate(
            problem,
            final_time=chunk,
            dt=dt,
            t0=0.0,
            history=history,
            method="RK45",
            rtol=rtol,
            atol=atol,
            backend=backend,
        )
        ys = np.asarray(traj.y, dtype=np.float64)
        ts = np.asarray(traj.t, dtype=np.float64)
        if not np.all(np.isfinite(ys)):
            raise ConvergenceError(
                f"{type(system).__name__}: DDE variational integration diverged "
                f"(chunk {c}); try a smaller dt or a looser tolerance."
            )

        new_base = np.empty((n_seg + 1, dim))
        new_dev = np.empty((n_seg + 1, k, dim))
        for d in range(dim):
            new_base[:, d] = np.interp(seg_t, ts, ys[:, d])
        for m in range(k):
            for d in range(dim):
                new_dev[:, m, d] = np.interp(seg_t, ts, ys[:, dim + m * dim + d])

        base_seg = new_base
        dev_seg, log_growth = _qr_segments(new_dev, k, dim)
        if c >= n_burn:
            log_sums += log_growth

    avg_time = n_avg * chunk  # == (n_chunks - n_burn) * chunk, the full final_time window
    exps = np.asarray(np.sort(log_sums / avg_time)[::-1])
    return exps


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
