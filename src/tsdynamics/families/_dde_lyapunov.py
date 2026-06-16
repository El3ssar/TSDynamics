"""Engine-based Lyapunov spectrum for delay systems (stream E-DDE-LYAP).

The Lyapunov spectrum of a DDE is the time-averaged logarithmic stretch of a set
of deviation *functions* carried along a trajectory.  Unlike an ODE — whose
tangent space is the finite ``R^dim`` — a DDE's tangent space is the
**infinite-dimensional history space** ``C([-τ, 0], R^dim)``, so a deviation is a
function on the delay window and there can be more exponents than state
components (``n_exp`` may exceed ``dim``).

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
is what lets ``n_exp`` exceed ``dim``.  The engine integrates one chunk of length
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

Validated against ``jitcdde_lyap`` on the five built-in DDEs (and a 2-D synthetic
DDE) — the Mackey–Glass leading exponent is positive, matching its
``known_lyapunov`` ``n_positive=1``; ``interp`` and ``jit`` agree bit-for-bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["dde_lyapunov_spectrum"]


def _build_extended_tape(system: Any, k: int) -> tuple[Any, list[Any], int]:
    """Lower the extended variational DDE of ``system`` with ``k`` deviations.

    Returns ``(tape, delay_slots, dim)``.  The tape is an ordinary RHS over
    ``dim·(k+1)`` state inputs plus the delay slots: the base system's slots
    (reused by the deviation equations) followed by one delayed-deviation slot
    per ``(deviation, base-slot)`` pair.  Parameters are folded to constants,
    matching :func:`tsdynamics.engine.compile.lower_dde`.
    """
    import symengine

    from tsdynamics.engine.compile import (
        DelaySlot,
        _is_past_y,
        _past_y_component_and_delay,
        lower_expressions,
    )
    from tsdynamics.engine.symbols import state_time_symbols
    from tsdynamics.families.continuous import _resolve_derivative_nodes

    y, t_sym = state_time_symbols()

    dim = system.dim
    exprs = list(type(system)._equations(y, t_sym, **system.params.as_dict()))
    if len(exprs) != dim:
        raise ValueError(f"_equations must return {dim} expressions, got {len(exprs)}")

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
        raise ValueError(f"{type(system).__name__}: no delayed terms — not a DDE")

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
        [_resolve_derivative_nodes(symengine.sympify(f[i]).diff(u[j])) for j in range(dim)]
        for i in range(dim)
    ]
    df_dd = [
        [_resolve_derivative_nodes(symengine.sympify(f[i]).diff(bdel[s])) for s in range(nb)]
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


def _seed_deviations(n_seg: int, n_exp: int, dim: int, grid: np.ndarray) -> np.ndarray:
    """Build ``n_exp`` distinct, QR-orthonormal initial deviation history segments.

    Distinct cosines over the delay window give linearly-independent seed
    functions even when ``n_exp > dim`` (the infinite-dimensional case); the
    burn-in then rotates them onto the leading covariant Lyapunov directions.
    """
    span = grid[-1] - grid[0]
    dev = np.zeros((n_seg + 1, n_exp, dim))
    for m in range(n_exp):
        wave = np.cos((m + 1) * np.pi * (grid - grid[0]) / (span + 1e-300))
        # Seed deviation m on state component m % dim with a distinct waveform, so
        # the k seed functions are linearly independent for any (n_exp, dim).
        dev[:, m, m % dim] = wave
    return _qr_segments(dev, n_exp, dim)[0]


def _qr_segments(dev: np.ndarray, n_exp: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """QR-orthonormalise ``k`` deviation history segments over the function space.

    ``dev`` has shape ``(n_seg+1, n_exp, dim)`` (point, deviation, component); each
    deviation's flattened ``(point, component)`` history is one column of an
    ``((n_seg+1)·dim, n_exp)`` matrix (its function-space representative), which is
    QR-decomposed.  The deviation axis must be the *column* axis and the component
    axis kept adjacent to the point axis (``transpose(0, 2, 1)``) — a plain
    ``reshape`` would interleave the deviation and component axes and scramble the
    columns for ``dim > 1``.  Returns the orthonormal segments (same shape) and
    ``log|diag R|`` (the per-deviation log growth over the chunk).
    """
    n_pts = dev.shape[0]
    mat = dev.transpose(0, 2, 1).reshape(n_pts * dim, n_exp)
    q, r = np.linalg.qr(mat)
    log_growth = np.log(np.abs(np.diag(r)))
    return q.reshape(n_pts, dim, n_exp).transpose(0, 2, 1), log_growth


def dde_lyapunov_spectrum(
    system: Any,
    *,
    n_exp: int = 1,
    final_time: float = 200.0,
    dt: float = 0.1,
    burn_in: float = 50.0,
    ic: Any = None,
    backend: str = "interp",
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> np.ndarray:
    """Estimate the ``n_exp`` leading Lyapunov exponents of a DDE on the engine.

    The function-space Benettin estimator described in the module docstring:
    integrate the extended variational DDE on the Rust engine in chunks of one
    delay window, QR-renormalise the deviation history segments each chunk, and
    time-average ``log|diag R|`` after burn-in.

    Parameters
    ----------
    system : DelaySystem
        The delay system.
    n_exp : int, default 1
        Number of leading exponents (may exceed ``dim`` — the tangent space is
        infinite-dimensional).
    final_time, dt, burn_in : float
        Averaging window, output/renormalisation step, and discarded transient.
        ``dt`` should divide the maximum delay for an exact base-history reuse.
    ic : array-like, optional
        Base initial state; pass the end state of a prior ``integrate`` so the
        run starts on the attractor (strongly recommended).
    backend : str, default "interp"
        ``"interp"`` or ``"jit"`` — the compiled engine.  ``"reference"`` is
        rejected (the engine has no pure-Python DDE integrator), matching DDE
        integration.
    rtol, atol : float
        Engine integration tolerances.

    Returns
    -------
    ndarray, shape (n_exp,)
        Exponents in descending order.
    """
    from scipy.interpolate import CubicSpline

    from tsdynamics.engine.problem import DDEProblem
    from tsdynamics.engine.run import integrate, resolve_backend

    n_exp = int(n_exp)
    if n_exp < 1:
        raise ValueError(f"n_exp must be >= 1, got {n_exp}")
    backend = resolve_backend(backend)
    if backend == "reference":
        raise NotImplementedError(
            "DDE Lyapunov has no pure-Python reference integrator; use "
            "backend='interp'/'jit' (the Rust engine), or the family's "
            "lyapunov_spectrum(backend='jitcdde') for the v2 path."
        )

    tape, slots, dim = _build_extended_tape(system, n_exp)
    max_delay = max(s.delay for s in slots)
    n_ext = dim * (n_exp + 1)

    # Chunk length = one delay window (minimal length giving a full past segment),
    # so the past window coincides with the engine output grid.
    chunk = max_delay
    n_seg = max(1, int(round(max_delay / dt)))
    grid = np.linspace(-max_delay, 0.0, n_seg + 1)

    # The deviation history segment has (n_seg+1)·dim sample dimensions, the ceiling
    # on the number of linearly-independent deviation functions it can carry.
    if n_exp > (n_seg + 1) * dim:
        raise ValueError(
            f"n_exp={n_exp} exceeds the delay-window resolution (n_seg+1)·dim="
            f"{(n_seg + 1) * dim}; decrease dt (finer window) or lower n_exp."
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
    dev_seg = _seed_deviations(n_seg, n_exp, dim, grid)

    n_chunks = max(1, int(round(final_time / chunk)))
    n_burn = min(n_chunks - 1, int(round(burn_in / chunk)))
    log_sums = np.zeros(n_exp)
    seg_t = chunk + grid  # absolute times of the next past window within [0, chunk]

    problem = DDEProblem(tape=tape, delay_slots=slots, ic=np.zeros(n_ext), system=system)

    for c in range(n_chunks):
        spl_base = [CubicSpline(grid, base_seg[:, d]) for d in range(dim)]
        spl_dev = [[CubicSpline(grid, dev_seg[:, m, d]) for d in range(dim)] for m in range(n_exp)]

        def history(s: float, spl_base=spl_base, spl_dev=spl_dev) -> np.ndarray:
            out = np.empty(n_ext)
            for d in range(dim):
                out[d] = spl_base[d](s)
            for m in range(n_exp):
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
            raise RuntimeError(
                f"{type(system).__name__}: DDE variational integration diverged "
                f"(chunk {c}); try a smaller dt or a looser tolerance."
            )

        new_base = np.empty((n_seg + 1, dim))
        new_dev = np.empty((n_seg + 1, n_exp, dim))
        for d in range(dim):
            new_base[:, d] = np.interp(seg_t, ts, ys[:, d])
        for m in range(n_exp):
            for d in range(dim):
                new_dev[:, m, d] = np.interp(seg_t, ts, ys[:, dim + m * dim + d])

        base_seg = new_base
        dev_seg, log_growth = _qr_segments(new_dev, n_exp, dim)
        if c >= n_burn:
            log_sums += log_growth

    avg_time = (n_chunks - n_burn) * chunk
    exps = np.sort(log_sums / avg_time)[::-1]
    return exps
