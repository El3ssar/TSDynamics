"""The universal per-system *dynamical* gate.

Every other registry-driven per-system check in this suite compares
``_equations`` against **itself**: the lowered tape against the symbolic form
(``test_ode_rhs_symbolic``), the interpreter against the JIT
(``test_xval_catalogue``), the reference oracle against the engine, the golden
tape hash against whatever was written (``test_equation_reference``).  Those are
excellent *consistency* gates and completely blind to a kernel that is
self-consistently **wrong**: a planted Lorenz typo (``-y(2)`` where the paper
says ``-y(1)``) collapses the butterfly to a fixed point and none of them can
tell.  (The golden hash does *notice* an edit to a system it already pins — but
it only reports "this tape changed", which is what a reviewed kernel change
looks like too, and it pins a **newly added** system to whatever was written the
first time.  It is a drift alarm, not a correctness oracle.)  The only universal
*dynamical* predicate anywhere was ``np.all(np.isfinite(traj.y))`` at
``final_time=2.0``, a bar that MultiChua clears from a ``U[0,1)^d`` start while
reaching ``|y| = 2.7e14`` and SprottP while reaching ``6.1e37``.

This module asserts, for **every** registered system, properties that are true
of the system it *claims to be*.  The claims come from the catalogue itself, so
a new system joins the gate with zero edits here.

Verified by re-planting the defects v6 actually shipped fixes for (monkeypatched
in-process, never on disk).  What fires, and where:

===================================== ============ ==================================
defect                                tier         verdict
===================================== ============ ==================================
Duffing, alpha/beta swapped           fast         ``ConvergenceError`` at t = 1.72
Baker, pre-v6 expanding form          fast         DEGENERATE (fixed point) + NOT
                                                   RECURRENT
Lorenz ``-y(2)`` typo                 slow         chaos claim: lambda_1 = -0.235
DoublePendulum, spurious factor 3     --           **not caught here** (see below)
===================================== ============ ==================================

The Lorenz typo is a fast-tier miss worth recording precisely: it turns the
attractor into a stable spiral, whose orbit is bounded and recurrent but still
*converging* — growth ratio 0.0070.  A fast-tier "still converging" floor would
have to sit between that and the tightest legitimate catalogue value (0.030,
KawczynskiStrizhak), a 4.3x window that is too narrow to ship as a gate, so the
claim predicate in the slow tier carries it instead (with a 235x margin).

The protocol
------------
Each system is run once, deterministically, into a **reference orbit**
(:func:`reference_orbit`) whose first half is discarded as transient.  Every
threshold is derived from the orbit's **own** scale — there is no hard-coded
``1e14``, because "bounded" is a statement about a system relative to itself,
and a catalogue that spans ``|y| ~ 0.1`` (Hénon-Heiles) to ``|y| ~ 10^4``
(Colpitts) has no single meaningful ceiling.

Four predicates run in the **fast** tier (the whole catalogue costs a few
seconds of integration; see ``test_the_gate_is_affordable``):

``computable``      the reference run completes and is finite;
``bounded``         the late orbit does not grow out of the early orbit's extent;
``non-degenerate``  it has not collapsed onto a fixed point or a 2-cycle, in
                    whole *or in part* (no coordinate is frozen while the rest
                    keep moving);
``revisiting``      the late orbit returns into the early orbit's neighbourhood.

Two claim-driven predicates run in the **slow** tier, because they need Lyapunov
exponents:

``chaotic``   a system the catalogue calls chaotic has a positive exponent;
``regular``   one it calls a limit cycle / quasiperiodic does not.

What this gate deliberately does **not** do
-------------------------------------------
It cannot see a mis-transcription that yields a *different but equally
well-behaved* system.  The v6 ``DoublePendulum`` defect (a spurious factor 3 on
the lower-arm gravitational torque) is exactly that: the wrong equations are
still a bounded, non-degenerate, recurrent, volume-preserving **Hamiltonian**
flow — just of a different pendulum.  Numerically: the symplectic-structure
residual ``max|M - Mᵀ|`` with ``M = -J·Df`` is 2.8e-10 for the defective form
and 3.6e-10 for the correct one, i.e. indistinguishable.  (Confirmed by
planting it: every predicate here passes, while
``test_catalogue_literature.py``'s two DoublePendulum checks — the compound-rod
Hamiltonian and the textbook normal modes — both fail.)  Defects of that class
need an independent re-derivation, which is what
``tests/test_equation_reference.py`` (layer 1) and
``tests/test_catalogue_literature.py`` exist for; this module is the third,
orthogonal layer, not a replacement for either.

Nor does it judge every system: 23 of the 177 carry only structural behaviour
tags, so no exponent claim is made about them.  They are named in
:data:`UNJUDGED_SYSTEMS` rather than skipped in silence.
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _sampling import DDE_HISTORIES, DYNAMICS_ICS, DYNAMICS_WINDOWS, SDE_SAMPLES

from tsdynamics import registry

# --------------------------------------------------------------------------- #
# The reference orbit
# --------------------------------------------------------------------------- #
#
# One deterministic run per system.  The windows are long enough that the first
# half is a genuine transient and the second half is on the attractor for every
# catalogue timescale (measured: the slowest catalogue transient, Rössler from a
# near-origin start, is settled well inside 50 time units).

#: ODE / SDE-free flow window: `T_FLOW` time units sampled every `DT_FLOW`.
T_FLOW = 100.0
DT_FLOW = 0.05

#: DDE window.  The method of steps lands on every sample, so `dt` bounds the
#: internal step; 400 units at dt=0.5 is 800 samples of settled dynamics.
T_DDE = 400.0
DT_DDE = 0.5

#: SDE window.  Short by design: the assertions on a stochastic path are about
#: the *process* staying where its drift puts it, and a long path of a
#: mean-reverting process adds nothing but wall-clock.
T_SDE = 20.0
DT_SDE = 0.01

#: Map orbit length.
STEPS_MAP = 4000

#: Radius of the generic starting ball for a system with no ``default_ic``.
#: Deliberately small: most catalogue flows are chaotic attractors whose basin
#: surrounds (but does not include) the origin, and a draw from ``U[0, 1)^dim``
#: — what the pre-v6 sweeps used — lands outside the basin of 9 of them.
IC_BALL_RADIUS = 0.1


def _seed_of(name: str) -> int:
    """A stable per-system seed (so a failure is reproducible from the name)."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def reference_ic(entry: Any) -> np.ndarray | None:
    """The deterministic starting point this gate holds ``entry`` to.

    Priority: the curated :data:`_sampling.DYNAMICS_ICS` override, then the
    class ``default_ic``, then a seeded draw from a ball of radius
    :data:`IC_BALL_RADIUS` about the origin.  Returns ``None`` for a map with
    neither an override nor a ``default_ic``, which means "let ``iterate`` draw
    and retry" — its random-IC retry is seeded from the constructor, so that is
    still deterministic.
    """
    override = DYNAMICS_ICS.get(entry.name)
    if override is not None:
        return np.asarray(override, dtype=float)
    default = getattr(entry.cls, "default_ic", None)
    if default is not None:
        return np.asarray(default, dtype=float)
    if entry.family == "map":
        return None
    dim = entry.cls().dim
    rng = np.random.default_rng(_seed_of(entry.name))
    return IC_BALL_RADIUS * (2.0 * rng.random(dim) - 1.0)


def _compute_reference_orbit(entry: Any) -> np.ndarray:
    """Integrate / iterate ``entry`` once under the gate's protocol."""
    system = entry.cls() if entry.family != "map" else entry.cls(seed=_seed_of(entry.name))
    ic = reference_ic(entry)
    if entry.family == "dde":
        return system.integrate(final_time=T_DDE, dt=DT_DDE, history=DDE_HISTORIES[entry.name]).y
    if entry.family == "sde":
        sample = SDE_SAMPLES[entry.name]
        return system.integrate(final_time=T_SDE, dt=DT_SDE, ic=sample["ic"], seed=sample["seed"]).y
    if entry.family == "map":
        return system.iterate(steps=STEPS_MAP, ic=ic, max_retries=20).y
    final_time, dt = DYNAMICS_WINDOWS.get(entry.name, (T_FLOW, DT_FLOW))
    return system.integrate(ic=ic, final_time=final_time, dt=dt).y


_ORBIT_CACHE: dict[str, np.ndarray] = {}


def reference_orbit(entry: Any) -> np.ndarray:
    """The cached reference orbit for ``entry`` (computed once per process).

    Several assertions share one run — the gate costs one integration per
    system, not one per predicate.
    """
    orbit = _ORBIT_CACHE.get(entry.name)
    if orbit is None:
        orbit = _compute_reference_orbit(entry)
        _ORBIT_CACHE[entry.name] = orbit
    return orbit


# --------------------------------------------------------------------------- #
# The statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OrbitStats:
    """Scale-free descriptors of a post-transient orbit.

    Every field is a **ratio**, so the same threshold means the same thing for
    Hénon-Heiles and for Colpitts.  ``early`` / ``late`` are the two halves of
    the post-transient orbit.
    """

    #: ``max|late - centre| / max|early - centre|``, ``centre`` = median(early).
    #: 1 for an orbit that stays put; unbounded for one that is leaving.
    growth: float
    #: ``max|y[n] - y[n-1]|`` per component, relative to that component's scale.
    #: Exactly 0 for a fixed point.
    step: float
    #: Same for a lag of 2.  0 for a fixed point *or* a period-2 cycle.
    stride: float
    #: Median distance from a late point to the nearest early point, in
    #: coordinates normalised by the early half's own extent and by the box
    #: diagonal.  Small when the orbit comes back; large when it has left.
    return_gap: float
    #: Number of components that actually move (the rest are frozen inputs).
    live_components: int
    #: Number of non-carrier components that never change **at all** over the
    #: whole post-transient orbit — an exact-bit test, so this is 0 for every
    #: catalogue system and non-zero only for a *partially* collapsed orbit.
    frozen_components: int
    #: Total number of non-carrier components (the denominator of the above).
    n_components: int
    #: Number of unbounded phase carriers excluded (see :func:`phase_components`).
    phase_carriers: int
    #: ``True`` when *every* component is a phase carrier, i.e. the system has
    #: no dynamics of its own beyond the phase it winds.
    all_phase: bool


#: How much a monotone component's increments may grow between the first and
#: the second half of the post-transient orbit and still count as a phase
#: carrier rather than as an escape.  A drive clock's increments are constant
#: and a winding angle's are merely bounded; a blow-up's grow without limit
#: (measured: >1e6 by the time the divergence guard fires), so the exact value
#: of this factor is not delicate.
PHASE_GROWTH_FACTOR = 4.0

#: How many late points are probed for a return, and how many early points they
#: are compared against.  Both are cost knobs for the dense distance block, not
#: statistics: subsampling the reference set can only *increase* the measured
#: gap, so the predicate stays on the conservative side of the truth.
RETURN_ANCHORS = 24
RETURN_BUDGET = 8_000_000


def phase_components(post: np.ndarray) -> np.ndarray:
    """Boolean mask of the **unbounded phase carriers** in a post-transient orbit.

    Two kinds of catalogue coordinate ramp without bound while the *system* is
    perfectly bounded, and neither is dynamics:

    * a **drive clock** — seventeen catalogue flows are non-autonomous systems
      written autonomously by carrying the forcing phase as a state with
      ``z' = omega`` (Duffing, every ``Forced*``, ArnoldWeb, the whole
      chaotic-advection family);
    * a **winding angle** — the advection systems live on a cylinder and carry
      ``theta`` on its universal cover (BlinkingRotlet's tracer circulates
      forever at ``r`` in [0.8, 0.91]).

    Left in, either pins the growth ratio near 3 for any window, dominates the
    return gap (successive laps are one ramp apart by construction), and — worst
    — would mask a collapse of the real state behind its own steady ticking.

    Detection is structural, never by name: a phase carrier is **strictly
    monotone** over the whole post-transient orbit *and* its increments are
    **steady** — the late-half mean within :data:`PHASE_GROWTH_FACTOR` of the
    early-half mean, in *both* directions.  The two-sided band is load-bearing:

    * increments may not **grow**, or an escaping component (which accelerates —
      measured >1e6 by the time the divergence guard fires) would be excluded
      and the boundedness predicate would go blind;
    * increments may not **shrink**, or a monotonically *converging* component
      (one settling onto an equilibrium, whose increments decay to zero) would
      be excluded and the degeneracy predicate would go blind.
    """
    if len(post) < 5:
        return np.zeros(post.shape[1], dtype=bool)
    delta = np.diff(post, axis=0)
    monotone = np.all(delta > 0.0, axis=0) | np.all(delta < 0.0, axis=0)
    half = len(delta) // 2
    tiny = np.finfo(float).tiny
    early = np.maximum(np.abs(delta[:half]).mean(axis=0), tiny)
    late = np.maximum(np.abs(delta[half:]).mean(axis=0), tiny)
    steady = (late <= PHASE_GROWTH_FACTOR * early) & (early <= PHASE_GROWTH_FACTOR * late)
    return monotone & steady


def dynamic_extent(orbit: np.ndarray) -> float:
    """How far the orbit's *state* ranges, relative to its own magnitude.

    Only used by the window-liveness guard, and only to answer "did the system
    actually move in this window?".  The drive clock is excluded (its range is
    the window length by construction and would swamp the answer); everything
    else, including a winding angle, counts as motion.
    """
    post = orbit[len(orbit) // 2 :]
    delta = np.diff(post, axis=0)
    monotone = np.all(delta > 0.0, axis=0) | np.all(delta < 0.0, axis=0)
    magnitude = np.maximum(np.abs(delta).mean(axis=0), np.finfo(float).tiny)
    clock = monotone & (delta.std(axis=0) <= 1e-6 * magnitude)
    cols = post if clock.all() else post[:, ~clock]
    span = cols.max(axis=0) - cols.min(axis=0)
    scale = np.maximum(np.abs(cols).max(axis=0), np.finfo(float).tiny)
    return float(np.max(span / scale))


def orbit_stats(orbit: np.ndarray) -> OrbitStats:
    """Reduce a reference orbit to its :class:`OrbitStats`."""
    full_post = orbit[len(orbit) // 2 :]
    carrier = phase_components(full_post)
    all_phase = bool(carrier.all())
    post = full_post if all_phase else full_post[:, ~carrier]

    half = len(post) // 2
    early, late = post[:half], post[half:]

    centre = np.median(early, axis=0)
    r_early = float(np.max(np.abs(early - centre)))
    r_late = float(np.max(np.abs(late - centre)))
    # A perfectly frozen early half makes any later motion unbounded growth.
    frozen_growth = np.inf if r_late > 0.0 else 0.0
    growth = r_late / r_early if r_early > 0.0 else frozen_growth

    scale = np.maximum(np.abs(post).max(axis=0), np.finfo(float).tiny)
    step = float(np.max(np.abs(post[1:] - post[:-1]) / scale))
    stride = float(np.max(np.abs(post[2:] - post[:-2]) / scale))

    # `step` / `stride` are maxima **over components**, so one healthy coordinate
    # hides any number of frozen ones.  Count the frozen ones separately, over
    # the whole post-transient orbit (more permissive than the early half used
    # for `live_components`, so a merely slow component is never mistaken for a
    # dead one) and by exact bit equality of the extremes.
    frozen = int(np.sum(post.max(axis=0) - post.min(axis=0) == 0.0))

    span = early.max(axis=0) - early.min(axis=0)
    live = span > 0.0
    n_live = int(live.sum())
    if n_live == 0:
        return_gap = float("inf")
    else:
        a = early[:, live] / span[live]
        b = late[:, live] / span[live]
        # The search is a dense RETURN_ANCHORS x |a| x n_live distance block, so
        # a 4608-component field (GrayScott) would allocate ~900 MB.  Thin the
        # reference half only when the block exceeds the budget — subsampling can
        # only make the gap *larger*, never smaller, so the predicate stays
        # conservative, but doing it unconditionally would inflate the gap of a
        # map (whose successive iterates are not neighbours) for no reason.
        budget_points = max(RETURN_ANCHORS, RETURN_BUDGET // (RETURN_ANCHORS * n_live))
        if len(a) > budget_points:
            a = a[np.linspace(0, len(a) - 1, budget_points).astype(int)]
        anchors = b[np.linspace(0, len(b) - 1, RETURN_ANCHORS).astype(int)]
        nearest = np.linalg.norm(anchors[:, None, :] - a[None, :, :], axis=2).min(axis=1)
        return_gap = float(np.median(nearest)) / np.sqrt(n_live)

    return OrbitStats(
        growth,
        step,
        stride,
        return_gap,
        n_live,
        frozen,
        int(post.shape[1]),
        int(carrier.sum()),
        all_phase,
    )


_STATS_CACHE: dict[str, OrbitStats] = {}


def reference_stats(entry: Any) -> OrbitStats:
    """The cached :class:`OrbitStats` of ``entry``'s reference orbit."""
    stats = _STATS_CACHE.get(entry.name)
    if stats is None:
        stats = orbit_stats(reference_orbit(entry))
        _STATS_CACHE[entry.name] = stats
    return stats


# --------------------------------------------------------------------------- #
# Thresholds — calibrated against the whole catalogue, with the margin recorded
# --------------------------------------------------------------------------- #

#: An orbit whose late half reaches more than this multiple of its early half's
#: extent is leaving, not orbiting.  Catalogue maximum **as shipped** (i.e. with
#: the :data:`_sampling.DYNAMICS_WINDOWS` extensions applied): 3.80
#: (SwingingAtwood), so this carries a 5.3x margin.  HastingsPowell scored 4.51
#: before its window extension and 1.05 after, which is the whole point of that
#: table.  A defective Duffing (alpha/beta swapped) does not even reach here —
#: it raises ConvergenceError first — while an escaping SprottP scores 2.7e5.
MAX_GROWTH = 20.0

#: Below this, consecutive samples are the same state to double precision and
#: the "orbit" is a fixed point (``step``) or a 2-cycle (``stride``).
#: Catalogue minimum ``step``: 5.15e-4 (BickleyJet), so this carries five
#: orders of margin.  The pre-v6 Baker collapsed to (0, 0) exactly: step = 0.
#: A *partial* collapse is covered separately by ``frozen_components``, which
#: needs no tolerance at all (exact bit equality, and 0 of 177 systems have a
#: frozen coordinate today).
DEGENERACY_TOL = 1e-9

#: A late point should land within this fraction of the early half's box
#: diagonal of some early point.  Catalogue maximum: **0.794**
#: (SwiftHohenberg, a coarsening pattern field), then 0.608 (Bouali2) and 0.491
#: (BickleyJet) — so the margin here is only **1.26x**, the thinnest of the four
#: fast predicates and the first number to re-derive if this ever fires on a
#: system nobody suspects.  (Recurrence is genuinely rarer in high dimension,
#: which is why the normalisation divides by ``sqrt(n_live)``.)  1.0 was chosen
#: over a roomier 2.0 deliberately: BlinkingRotlet's genuine long transient
#: measured 1.21, and the predicate is meant to have caught it.
MAX_RETURN_GAP = 1.0

#: A growth ratio *below* this means the orbit is still collapsing inwards —
#: the transient has not finished, so the window is measuring the approach and
#: not the attractor.  Used **only** by the window-liveness guard, never as a
#: universal predicate: the catalogue minimum is 0.030 (KawczynskiStrizhak),
#: which is too close to be a safe gate, whereas a false "the override is still
#: needed" verdict merely keeps a documented table entry.
SETTLED_GROWTH_FLOOR = 0.01

#: A system the catalogue calls chaotic must clear this leading exponent, in the
#: system's own inverse time units.  Weakly chaotic catalogue members sit an
#: order of magnitude above it (HastingsPowell 0.010, ItikBanksTumor 0.013).
CHAOS_LAMBDA_FLOOR = 1e-3

#: A system the catalogue calls regular (limit cycle / quasiperiodic) must stay
#: below this.  A finite-time estimate of an exactly-zero exponent is noisy;
#: this is two orders above the floor above, so the two predicates cannot both
#: be satisfied by the same number.
REGULAR_LAMBDA_CEILING = 0.1


# --------------------------------------------------------------------------- #
# Carve-outs — every exclusion is named, reasoned and liveness-checked
# --------------------------------------------------------------------------- #

#: Systems exempt from :data:`MAX_GROWTH`, because unbounded growth is the
#: documented behaviour rather than a defect.  Guarded by
#: ``test_boundedness_carve_outs_are_all_live``: an entry that would now pass
#: the predicate fails the suite, so this table cannot rot into a blanket skip.
#: Empty today: even ``GeometricBrownianMotion``, the one catalogue system that
#: is unbounded by construction, grows only 1.27x over its reference window at
#: the shipped drift, so it is held to the predicate like everything else.  Kept
#: as the documented extension point — raising that drift would need an entry
#: here rather than a loosened threshold.
UNBOUNDED_BY_CONSTRUCTION: dict[str, str] = {}

#: Systems allowed to settle onto a fixed point or a 2-cycle, or to carry a
#: permanently frozen coordinate (a conserved quantity held as a state, say).
#: Empty: no catalogue system currently does either — 0 of 177 have so much as
#: one frozen component — and the pre-v6 Baker (which collapsed outright) was a
#: bug.
#: Kept as the documented extension point so a future genuinely-equilibrium
#: system is added deliberately rather than by loosening the threshold.
SETTLES_TO_EQUILIBRIUM: dict[str, str] = {}

#: Systems whose orbit is a **transport** problem rather than an attractor: the
#: tracer travels and never comes back, so *every* component is a phase carrier
#: by design and "not all-phase" is not a claim the system makes.  Exempt from
#: that one clause of the degeneracy predicate only — bounded, non-degenerate
#: and revisiting are still asserted (on the phase coordinates themselves, which
#: is the honest reading for a transport problem).  Liveness-checked by
#: ``test_transport_carve_outs_are_all_live``.
PURE_TRANSPORT: dict[str, str] = {
    "BickleyJet": (
        "a passive tracer in a zonal jet, tagged 'quasiperiodic transport': it "
        "is advected downstream forever, so both spatial coordinates drift "
        "monotonically alongside the clock."
    ),
}


# --------------------------------------------------------------------------- #
# The behaviour claim
# --------------------------------------------------------------------------- #
#
# The catalogue's own claim about each system is the curated ``behavior`` tag in
# docs/_tooling/editorial.json — a reviewed, per-system, single-word statement
# that is far more reliable than parsing "chaotic" out of prose (SprottJerk's
# docstring says "reaches chaos ... around mu = 2.017", which is a statement
# about a bifurcation, not about the defaults).  Every tag in the file must be
# classified below; ``test_every_behaviour_tag_is_classified`` fails on a new
# one rather than letting it fall silently into "no claim".

_EDITORIAL_PATH = Path(__file__).resolve().parents[1] / "docs" / "_tooling" / "editorial.json"

#: Tags asserting a positive leading Lyapunov exponent.
CHAOS_TAGS = frozenset(
    {
        "chaotic",
        "hyperchaotic",
        "chaotic advection",
        "chaotic cycles",
        "cycling chaos",
        "mixed-mode chaotic",
        "mixing",
        "intermittent",
    }
)

#: Tags asserting a *non*-positive leading exponent (regular motion).
REGULAR_TAGS = frozenset(
    {
        "limit cycle",
        "cycles",
        "quasiperiodic",
        "quasiperiodic transport",
        "oscillatory",
        "relaxation",
        "mode-locking",
        "mean-reverting",
    }
)

#: Tags that describe *structure* rather than the character of the motion, and
#: so make no claim about the exponent on their own.  Listed explicitly so the
#: coverage guard can tell "deliberately neutral" from "nobody looked".
NEUTRAL_TAGS = frozenset(
    {
        "attractor",
        "bistable",
        "bursting",
        "circuit",
        "climate",
        "conservative",
        "cyclically symmetric",
        "dissipative",
        "excitable",
        "forced",
        "mixed",
        "multiplicative",
        "non-smooth",
        "normal form",
        "patterns",
        "period-doubling",
        "reversals",
        "spatiotemporal",
        "switching",
        "synchronization",
    }
)

CLAIM_CHAOTIC = "chaotic"
CLAIM_REGULAR = "regular"
CLAIM_NONE = "no-claim"

#: Per-system overrides of the tag-derived claim, each with its reason.  These
#: are the systems whose editorial tag is about something other than the
#: character of the default orbit; the gate says so here instead of quietly
#: asserting the wrong thing.
CLAIM_OVERRIDES: dict[str, tuple[str, str]] = {
    "LorenzCoupled": (
        CLAIM_CHAOTIC,
        "tagged 'synchronization' — the phenomenon of interest — but the "
        "synchronised state of two coupled Lorenz systems is still the Lorenz "
        "attractor, and the default coupling does not fully synchronise them.",
    ),
}


def _load_editorial() -> dict[str, dict]:
    """The per-system editorial records (the catalogue's own claims)."""
    return dict(json.loads(_EDITORIAL_PATH.read_text(encoding="utf-8"))["systems"])


def behaviour_tags(name: str, editorial: dict[str, dict] | None = None) -> frozenset[str]:
    """The normalised ``behavior`` tag set for ``name``.

    The field is a list for most systems and a bare (sometimes comma-joined)
    string for a few — ``"forced, chaotic"`` — so both spellings are normalised
    to a set of individual tags here.
    """
    record = (editorial if editorial is not None else _load_editorial()).get(name, {})
    raw = record.get("behavior") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    return frozenset(tag for tag in raw if tag)


def behaviour_claim(name: str, editorial: dict[str, dict] | None = None) -> str:
    """Classify what the catalogue claims ``name`` does.

    Returns :data:`CLAIM_CHAOTIC`, :data:`CLAIM_REGULAR` or :data:`CLAIM_NONE`.
    A chaos tag wins over a regularity tag (``"chaotic, bursting"`` is a claim
    of chaos), and an explicit :data:`CLAIM_OVERRIDES` entry wins over both.
    """
    if name in CLAIM_OVERRIDES:
        return CLAIM_OVERRIDES[name][0]
    tags = behaviour_tags(name, editorial)
    if tags & CHAOS_TAGS:
        return CLAIM_CHAOTIC
    if tags & REGULAR_TAGS:
        return CLAIM_REGULAR
    return CLAIM_NONE


# --------------------------------------------------------------------------- #
# Fast tier — the universal dynamical predicates
# --------------------------------------------------------------------------- #


def test_reference_orbit_is_a_bounded_recurrent_attractor(system_entry) -> None:
    """All four universal predicates, on one deterministic run per system.

    They are checked together rather than as four parametrized tests purely for
    cost: the suite runs under ``pytest -n auto``, which distributes by *test*,
    so four tests per system would integrate each system up to four times on
    four different workers.  One test per system means one integration, and
    every violated predicate is reported at once instead of only the first.

    **computable** — the run completes and stays finite.  This is what the
    pre-v6 Duffing (``alpha`` and ``beta`` swapped between the linear and the
    cubic term, inverting the potential into an unbounded quartic) fails
    outright::

        ConvergenceError: integration diverged before reaching the final time:
        state magnitude reached 1.0064e+150 at t = 12.63

    **bounded** — "finite" is not "bounded": SprottF reaches ``|y| = 1.2e19``
    and MultiChua ``5.1e14`` from an off-basin start and both are perfectly
    finite.  The predicate compares the late half of the orbit to the early half
    of the *same* orbit, so it is scale-free and needs no per-system ceiling.

    **non-degenerate** — the orbit has not collapsed onto a fixed point or a
    2-cycle.  This is the pre-v6 Baker defect, which no gate saw: written as an
    expanding map (``|det J| = 4``), float doubling drained its mantissa and
    *every* orbit landed on ``(0, 0)`` within ~53 iterations, with shape,
    finiteness, tape hash and interp-vs-jit agreement all still green.
    The clause also covers a **partial** collapse, which the other three
    predicates structurally cannot see because they are maxima over components:
    freezing one lattice site of Lorenz-96 (``x_0' = 0``, a dropped-term typo in
    a system that carries no exponent claim, so this is its only universal
    check) leaves growth 1.19, step 0.38 and return gap 0.11 — indistinguishable
    from health — while 1 of the 20 coordinates never moves a bit.

    **revisiting** — a trajectory on an attractor keeps returning to where it
    has already been; one that is escaping, or drifting along an unbounded
    direction that the boundedness ratio happens to tolerate, does not.
    """
    orbit = reference_orbit(system_entry)
    name = system_entry.name
    assert orbit.ndim == 2 and len(orbit) > 4, f"{name}: degenerate orbit array"
    assert np.all(np.isfinite(orbit)), f"{name}: the reference orbit contains non-finite values"

    stats = reference_stats(system_entry)
    problems: list[str] = []

    if name not in UNBOUNDED_BY_CONSTRUCTION and stats.growth > MAX_GROWTH:
        problems.append(
            f"NOT BOUNDED: the orbit is leaving, not orbiting — its late half reaches "
            f"{stats.growth:.4g}x the extent of its early half (limit {MAX_GROWTH}). "
            f"Either the kernel escapes to infinity, or the reference initial "
            f"condition is outside the basin (add one to _sampling.DYNAMICS_ICS)."
        )

    if name not in SETTLES_TO_EQUILIBRIUM:
        if stats.all_phase and name not in PURE_TRANSPORT:
            problems.append(
                "DEGENERATE: every component is an unbounded phase carrier — the "
                "system has no dynamics of its own on this orbit (it is ramping, not "
                "orbiting; the reference IC may be outside the physical domain)."
            )
        if stats.step <= DEGENERACY_TOL:
            problems.append(
                f"DEGENERATE: the orbit has collapsed to a fixed point — consecutive "
                f"samples agree to {stats.step:.3g} of their own scale (tolerance "
                f"{DEGENERACY_TOL})."
            )
        elif stats.stride <= DEGENERACY_TOL:
            problems.append(
                f"DEGENERATE: the orbit has collapsed to a 2-cycle — samples two apart "
                f"agree to {stats.stride:.3g} of their own scale (tolerance "
                f"{DEGENERACY_TOL})."
            )
        # ...and, when the orbit as a whole is still moving, whether *part* of
        # it has died.  (A total collapse freezes every component too, but that
        # is already reported above and saying it twice only obscures it.)
        if stats.frozen_components and stats.step > DEGENERACY_TOL:
            problems.append(
                f"PARTIALLY DEGENERATE: {stats.frozen_components} of "
                f"{stats.n_components} component(s) never change at all over the whole "
                f"post-transient orbit (not one bit). The state has partially "
                f"collapsed — a dropped or zeroed coupling term freezes a coordinate "
                f"while the rest keep moving, which every other predicate here reads "
                f"as healthy because they are maxima over components."
            )

    if name not in UNBOUNDED_BY_CONSTRUCTION and stats.return_gap > MAX_RETURN_GAP:
        problems.append(
            f"NOT RECURRENT: the orbit does not return to where it has been — a "
            f"typical late point sits {stats.return_gap:.4g} box-diagonals from the "
            f"nearest earlier point (limit {MAX_RETURN_GAP}); it is on a transient or "
            f"escaping, not on an attractor."
        )

    assert not problems, (
        f"{name} is not the attractor it claims to be:\n  "
        + "\n  ".join(problems)
        + f"\n  (stats: {stats})"
    )


# --------------------------------------------------------------------------- #
# Fast tier — the claim bookkeeping (no integration)
# --------------------------------------------------------------------------- #


def test_the_claim_source_covers_the_whole_catalogue() -> None:
    """``editorial.json`` exists and has a record for every registered system.

    The behaviour tags are this gate's *only* source of truth about what each
    system claims to do, and they live outside the package.  If that file were
    moved, renamed or left behind by a new system, the claim would silently
    become "no claim" and a hundred Lyapunov assertions would evaporate without
    a single test turning red.  So the coupling is asserted rather than assumed.
    """
    assert _EDITORIAL_PATH.exists(), (
        f"the behaviour-claim source {_EDITORIAL_PATH} is missing — this gate's "
        f"Lyapunov predicates have nothing to check against"
    )
    editorial = _load_editorial()
    catalogue = {entry.name for entry in registry.all_systems()}
    unlisted = sorted(catalogue - set(editorial))
    assert not unlisted, (
        f"{unlisted} have no editorial record, so this gate makes no claim about "
        f"them; add a `behavior` tag in {_EDITORIAL_PATH.name}"
    )
    claimed = sum(1 for name in catalogue if behaviour_claim(name, editorial) != CLAIM_NONE)
    assert claimed >= 140, (
        f"only {claimed} of {len(catalogue)} systems carry an exponent claim (was "
        f"154); the classifier or the tag vocabulary has regressed"
    )


def test_every_behaviour_tag_is_classified() -> None:
    """Every editorial behaviour tag is chaos / regular / deliberately neutral.

    Without this, adding a tag the classifier does not know silently demotes its
    systems to "no claim" and the Lyapunov gate stops looking at them.
    """
    editorial = _load_editorial()
    seen: set[str] = set()
    for name in editorial:
        seen |= behaviour_tags(name, editorial)
    known = CHAOS_TAGS | REGULAR_TAGS | NEUTRAL_TAGS
    unknown = sorted(seen - known)
    assert not unknown, (
        f"unclassified behaviour tag(s) {unknown} in editorial.json — add each to "
        f"CHAOS_TAGS, REGULAR_TAGS or NEUTRAL_TAGS (neutral = 'makes no claim "
        f"about the leading exponent')"
    )
    stale = sorted(known - seen - {"non-smooth"})
    assert not stale, f"behaviour tag(s) {stale} are classified here but no longer used"


def test_the_gate_names_the_systems_it_cannot_judge() -> None:
    """The unclassified systems are *reported*, not silently skipped.

    A gate that quietly drops the systems it has no claim for is the failure
    mode this whole module exists to fix, so the roster is asserted rather than
    left implicit: if it changes, this test prints the new one.  Coverage today
    is 154 of 177 systems carrying an exponent claim (135 chaotic, 19 regular);
    of those, 147 are measured here and the other 7 are named in
    :data:`UNMEASURED_CLAIMS`.
    """
    editorial = _load_editorial()
    unclaimed = sorted(
        e.name for e in registry.all_systems() if behaviour_claim(e.name, editorial) == CLAIM_NONE
    )
    assert unclaimed == sorted(UNJUDGED_SYSTEMS), (
        "the set of systems this gate makes no exponent claim about changed.\n"
        f"  now:      {unclaimed}\n"
        f"  expected: {sorted(UNJUDGED_SYSTEMS)}\n"
        "Each of these carries only structural editorial tags (conservative / "
        "bursting / patterns / ...), which say nothing about the sign of the "
        "leading exponent. If one of them *is* chaotic or regular, tag it in "
        "docs/_tooling/editorial.json (preferred) or add a CLAIM_OVERRIDES entry."
    )


#: The systems whose editorial tags are purely structural, so this gate asserts
#: the four universal predicates on them but makes no Lyapunov claim.  Pinned so
#: the roster can only shrink or grow deliberately.
UNJUDGED_SYSTEMS: list[str] = [
    # `conservative`: a Hamiltonian / volume-preserving system is chaotic on some
    # orbits and integrable on others *at the same parameters* — the sign of the
    # exponent is a property of the initial condition, not of the system.
    "ArnoldWeb",
    "Chirikov",
    "NoseHoover",
    "NuclearQuadrupole",
    "SprottA",
    "SprottTorus",
    # `attractor`: the decorative strange-attractor maps, tagged for what they
    # draw rather than for how they stretch.
    "Bedhead",
    "DeJong",
    "Hopalong",
    "Pickover",
    "Svensson",
    # `bursting` / `excitable` / `switching`: relaxation-type dynamics that can
    # be periodic bursting or chaotic bursting at the same defaults.
    "CaTwoPlus",
    "DoubleWell",
    "ExcitableCell",
    "ForcedFitzHughNagumo",
    # `forced`: the tag names the drive, not the response.
    "ForcedBrusselator",
    "ForcedVanDerPol",
    "StickSlipOscillator",
    # `patterns` / `spatiotemporal`: method-of-lines fields, where the leading
    # exponent is a statement about a discretisation as much as about the PDE.
    "GrayScott",
    "KuramotoSivashinsky",
    "Lorenz96",
    "SwiftHohenberg",
    # `multiplicative`: a stochastic process has no deterministic tangent flow.
    "GeometricBrownianMotion",
]


@pytest.mark.slow
def test_boundedness_carve_outs_are_all_live() -> None:
    """Every :data:`UNBOUNDED_BY_CONSTRUCTION` entry still needs its exemption.

    A carve-out that has quietly become unnecessary is a hole in the gate, so
    each one is re-checked: if the system now satisfies the predicate it is
    exempt from, the exemption must go.
    """
    for name, reason in UNBOUNDED_BY_CONSTRUCTION.items():
        entry = registry.get(name)
        stats = reference_stats(entry)
        assert stats.growth > MAX_GROWTH, (
            f"{name} is exempt from the boundedness predicate ({reason}) but now "
            f"passes it (growth {stats.growth:.4g} <= {MAX_GROWTH}); drop the "
            f"UNBOUNDED_BY_CONSTRUCTION entry."
        )


@pytest.mark.slow
def test_transport_carve_outs_are_all_live() -> None:
    """Every :data:`PURE_TRANSPORT` entry still needs its exemption."""
    for name, reason in PURE_TRANSPORT.items():
        stats = reference_stats(registry.get(name))
        assert stats.all_phase, (
            f"{name} is exempt from the all-phase clause ({reason}) but its orbit "
            f"now has {stats.live_components} genuinely dynamic component(s); drop "
            f"the PURE_TRANSPORT entry."
        )


@pytest.mark.slow
def test_reference_ic_overrides_are_all_live() -> None:
    """Every :data:`_sampling.DYNAMICS_ICS` entry is still needed.

    An override that the generic rule would now handle is dead weight that hides
    a change in the system, so each is checked against the generic starting
    point: if the generic point reaches the same verdict, the override goes.
    "Verdict" here includes the behaviour claim, not only the four fast
    predicates — two of these overrides (SprottE, ItikBanksTumor) exist precisely
    because the generic ball lands on a bounded, recurrent, non-degenerate orbit
    that is nevertheless the *wrong* one (an invariant axis and a stable fixed
    point respectively), which only the exponent can tell you.
    """
    editorial = _load_editorial()
    still_needed: list[str] = []
    for name in DYNAMICS_ICS:
        entry = registry.get(name)
        system = entry.cls()
        rng = np.random.default_rng(_seed_of(name))
        generic = IC_BALL_RADIUS * (2.0 * rng.random(system.dim) - 1.0)
        final_time, dt = DYNAMICS_WINDOWS.get(name, (T_FLOW, DT_FLOW))
        try:
            stats = orbit_stats(system.integrate(ic=generic, final_time=final_time, dt=dt).y)
            needed = (
                stats.growth > MAX_GROWTH
                or stats.return_gap > MAX_RETURN_GAP
                or stats.all_phase
                or stats.step <= DEGENERACY_TOL
            )
            if not needed and behaviour_claim(name, editorial) == CLAIM_CHAOTIC:
                lam = float(
                    entry.cls().lyapunov_spectrum(
                        k=1, ic=generic, dt=DT_FLOW, burn_in=50.0, final_time=400.0
                    )[0]
                )
                needed = lam <= CHAOS_LAMBDA_FLOOR
        except Exception:  # noqa: BLE001 — any failure means the override is needed
            needed = True
        if needed:
            still_needed.append(name)
    assert sorted(still_needed) == sorted(DYNAMICS_ICS), (
        "these DYNAMICS_ICS overrides are no longer needed — the generic "
        f"starting ball now works: {sorted(set(DYNAMICS_ICS) - set(still_needed))}"
    )


@pytest.mark.slow
def test_reference_windows_are_all_live() -> None:
    """Every :data:`_sampling.DYNAMICS_WINDOWS` entry is still needed.

    Same liveness rule as the IC overrides: a system that now settles inside the
    default window must lose its extension, so the table cannot quietly become a
    way of integrating past a defect.
    """
    for name, window in DYNAMICS_WINDOWS.items():
        entry = registry.get(name)
        default_run = entry.cls().integrate(ic=reference_ic(entry), final_time=T_FLOW, dt=DT_FLOW).y
        stats = orbit_stats(default_run)
        predicates_hold = (
            SETTLED_GROWTH_FLOOR <= stats.growth <= MAX_GROWTH
            and stats.return_gap <= MAX_RETURN_GAP
            and not (stats.all_phase and name not in PURE_TRANSPORT)
        )
        # ...and did the system move at all?  A window can also be too short
        # simply because the dynamics are slow in the system's own units
        # (BickleyJet's jet speed is 6.3e-5), in which case every predicate is
        # trivially satisfied by an orbit that has barely left its start.
        explored = dynamic_extent(default_run) >= 0.2 * dynamic_extent(reference_orbit(entry))
        assert not (predicates_hold and explored), (
            f"{name} has a custom reference window {window} but already settles "
            f"inside the default ({T_FLOW}, {DT_FLOW}): growth {stats.growth:.4g}, "
            f"return gap {stats.return_gap:.4g}, relative extent "
            f"{dynamic_extent(default_run) / dynamic_extent(reference_orbit(entry)):.3g}; "
            f"drop the DYNAMICS_WINDOWS entry."
        )


def test_known_lyapunov_metadata_agrees_with_the_behaviour_claim() -> None:
    """A declared ``known_lyapunov`` cannot contradict the behaviour tag.

    Free (pure metadata), and it catches the two halves of the catalogue's
    claim drifting apart — a system re-tagged ``limit cycle`` while still
    declaring ``n_positive: 1``, or vice versa.
    """
    editorial = _load_editorial()
    problems: list[str] = []
    for entry in registry.all_systems():
        meta = entry.known_lyapunov
        if not meta:
            continue
        claim = behaviour_claim(entry.name, editorial)
        if "n_positive" in meta:
            n_pos = int(meta["n_positive"])
        elif "spectrum" in meta:
            n_pos = int(np.sum(np.asarray(meta["spectrum"], dtype=float) > 1e-6))
        else:
            continue
        if claim == CLAIM_CHAOTIC and n_pos < 1:
            problems.append(f"{entry.name}: tagged chaotic but known_lyapunov has {n_pos} positive")
        if claim == CLAIM_REGULAR and n_pos > 0:
            problems.append(f"{entry.name}: tagged regular but known_lyapunov has {n_pos} positive")
    assert not problems, "known_lyapunov contradicts the behaviour tag:\n  " + "\n  ".join(problems)


# --------------------------------------------------------------------------- #
# Slow tier — the claim actually holds
# --------------------------------------------------------------------------- #

#: Systems excluded from the exponent sweep, each with the reason.  Flows only:
#: a DDE spectrum needs the (expensive) history-space estimator and an SDE has
#: no deterministic tangent flow at all.
#:
#: Kept deliberately small, and **liveness-checked** by
#: :func:`test_lyapunov_exclusions_are_all_live`: this is the one table that can
#: make a *claim-carrying* system disappear from the gate, so an entry that is
#: merely inconvenient rather than infeasible is a hole.  Three earlier entries
#: were removed on measurement (deterministic, repeated twice, this machine):
#: ``Chua`` +0.4235 in **0.2 s** (the cheapest member of the table, not the most
#: expensive — its "piecewise-linear" reason was never a cost),
#: ``BelousovZhabotinsky`` +25.63 in **8.0 s** (the quoted ~25 s did not
#: reproduce), and ``MacArthur`` -1.55e-4 in 24 s — which is not a cost finding
#: at all but a *discrepancy*, now recorded in :data:`CHAOS_CLAIM_UNRESOLVED`
#: where it is reported on every run instead of hidden behind an exclusion.
LYAPUNOV_EXCLUDE: dict[str, str] = {
    "GrayScott": "high-dimensional method-of-lines field (dim 4608)",
    "SwiftHohenberg": "high-dimensional method-of-lines field (dim 1024)",
    "KuramotoSivashinsky": "high-dimensional method-of-lines field",
}


def _leading_exponent(entry: Any) -> float:
    """The leading Lyapunov exponent of ``entry`` from the gate's reference IC."""
    system = entry.cls()
    ic = reference_ic(entry)
    if entry.family == "map":
        return float(system.lyapunov_spectrum(k=1, steps=20_000, ic=ic)[0])
    return float(
        system.lyapunov_spectrum(k=1, ic=ic, dt=DT_FLOW, burn_in=50.0, final_time=400.0)[0]
    )


# --------------------------------------------------------------------------- #
# Recorded discrepancies — the gate's findings, kept visible rather than hidden
# --------------------------------------------------------------------------- #
#
# These are catalogue systems whose documented behaviour the measurement does
# NOT reproduce at the shipped defaults, from the gate's reference initial
# condition.  They are recorded as **strict xfails**, not skips: the run reports
# them every time, and the moment one starts agreeing (a fixed kernel, a repaired
# default, a corrected tag) the suite fails and demands this table be updated.
# The measured exponent is quoted so the next reader does not have to re-derive
# it.  Widening CHAOS_LAMBDA_FLOOR to swallow them was the alternative, and is
# exactly the kind of accommodation that made the old gates vacuous.
#
# Every entry below was cross-checked over a 5x longer averaging window and from
# three further initial conditions before being written down; where a different
# start *does* find the documented behaviour, the fix was an entry in
# _sampling.DYNAMICS_ICS instead (SprottE and ItikBanksTumor went that way).

CHAOS_CLAIM_UNRESOLVED: dict[str, str] = {
    "DequanLi": (
        "documented chaotic; measured lambda_1 = +2.9e-4 (gate window) and "
        "+1.5e-4 over a 5x window, from four different starts — a torus, not chaos"
    ),
    "Robinson": (
        "documented chaotic; measured lambda_1 = +3.9e-4, and its sign flips with "
        "the initial condition over a 5x window (-6.5e-4 to +8.8e-4)"
    ),
    "CaTwoPlusQuasiperiodic": (
        "tagged chaotic; measured lambda_1 = -1.7e-3 and |lambda_1| < 6e-4 over a "
        "5x window from three starts. The system's own name says quasiperiodic, so "
        "the editorial tag is the likelier error"
    ),
    "WindmiReduced": (
        "documented chaotic; measured lambda_1 = -0.19 (gate start) and -2.7e-8 "
        "from the docs page's own [1, 1, 1] — a limit cycle either way"
    ),
    "CoevolvingPredatorPrey": (
        "documented chaotic; measured lambda_1 = -3.2e-3. Over a 5x window it "
        "reads +2.9e-3 but the orbit is by then escaping (growth 5.8, return gap "
        "39), so the positive value is a transient, not an attractor"
    ),
    "KawczynskiStrizhak": (
        "documented chaotic + bursting; measured lambda_1 = -9.4e-3, and -1.3e-3 "
        "over a 5x window; the four starts tried scatter over +-3e-3 around zero"
    ),
    "MacArthur": (
        "documented chaotic; measured lambda_1 = -1.55e-4 (deterministic, repeated). "
        "Previously invisible: it sat in LYAPUNOV_EXCLUDE on a cost argument, so the "
        "gate never asked. It costs ~24 s (the stiff variational lowering) and that "
        "is worth paying to keep the finding on the board rather than in a table"
    ),
    "Bogdanov": (
        "documented chaotic; measured lambda_1 = +2.9e-4 at 20k iterates and "
        "+5.1e-5 at 200k — converging to zero. At the shipped eps = mu = 0 the map "
        "is the conservative Bogdanov map, whose orbits are KAM tori for most starts"
    ),
}

REGULAR_CLAIM_UNRESOLVED: dict[str, str] = {
    "Circle": (
        "tagged mode-locking; measured lambda_1 = +1.085 (stable to 200k iterates). "
        "The shipped k = 5.7 is far above the critical k = 1, where the circle map "
        "is non-invertible and chaotic — the tag describes k < 1"
    ),
    "MaynardSmith": (
        "tagged cycles; measured lambda_1 = +0.130 (stable to 200k iterates) — "
        "chaotic at the shipped a = 0.87, b = 0.75"
    ),
}


def _exponent_params(claim: str, unresolved: dict[str, str]) -> list[Any]:
    """Parametrization for one claim, xfailing the recorded discrepancies."""
    editorial = _load_editorial()
    params = []
    for entry in registry.all_systems():
        if entry.family not in ("ode", "map"):
            continue
        if entry.name in LYAPUNOV_EXCLUDE or behaviour_claim(entry.name, editorial) != claim:
            continue
        marks = []
        if entry.name in unresolved:
            marks.append(
                pytest.mark.xfail(strict=True, reason=f"{entry.name}: {unresolved[entry.name]}")
            )
        params.append(pytest.param(entry, id=entry.name, marks=marks))
    return params


_CHAOTIC_PARAMS = _exponent_params(CLAIM_CHAOTIC, CHAOS_CLAIM_UNRESOLVED)
_REGULAR_PARAMS = _exponent_params(CLAIM_REGULAR, REGULAR_CLAIM_UNRESOLVED)


@pytest.mark.slow
@pytest.mark.parametrize("entry", _CHAOTIC_PARAMS)
def test_systems_claimed_chaotic_have_a_positive_exponent(entry) -> None:
    """A system the catalogue calls chaotic must actually stretch.

    The counterpart of ``test_known_values``' exact ``n_positive`` count, but
    driven by the *catalogue's* claim rather than by opt-in metadata, so it
    covers a hundred systems instead of twenty-one.  A kernel that has been
    mis-transcribed into a limit cycle — or into an equilibrium, which is what
    the planted Lorenz ``-y(2)`` typo produces — fails here even though every
    consistency gate stays green.
    """
    lam = _leading_exponent(entry)
    assert lam > CHAOS_LAMBDA_FLOOR, (
        f"{entry.name} is documented as chaotic but its leading Lyapunov exponent "
        f"is {lam:.5g} (floor {CHAOS_LAMBDA_FLOOR}). Either the default "
        f"parameters/initial condition miss the chaotic regime, or the kernel is wrong."
    )


@pytest.mark.slow
@pytest.mark.parametrize("entry", _REGULAR_PARAMS)
def test_systems_claimed_regular_have_no_positive_exponent(entry) -> None:
    """A system the catalogue calls a limit cycle / quasiperiodic must not stretch.

    The other direction matters just as much: a "limit cycle" that turns out to
    have a positive exponent is either mislabelled or mis-transcribed.
    """
    lam = _leading_exponent(entry)
    assert lam < REGULAR_LAMBDA_CEILING, (
        f"{entry.name} is documented as regular (limit cycle / quasiperiodic) but "
        f"its leading Lyapunov exponent is {lam:.5g} (ceiling "
        f"{REGULAR_LAMBDA_CEILING}) — it is behaving chaotically."
    )


#: Systems that *do* carry an exponent claim but which the sweep above does not
#: measure, with the reason.  The roster exists because "makes a claim" and "is
#: checked" are two different sets, and the gap between them is precisely where
#: a gate goes quietly vacuous — the failure mode this module was written to
#: end.  ``test_the_gate_names_the_claims_it_does_not_measure`` pins it.
UNMEASURED_CLAIMS: dict[str, str] = {
    # DDEs: the tangent space is the infinite-dimensional history space, so the
    # exponent needs DelaySystem.lyapunov_spectrum (the history-space Benettin
    # estimator, `families/_dde_lyapunov.py`) rather than the flow path used
    # above, and it costs an order of magnitude more.  Only MackeyGlass is
    # covered elsewhere (`known_lyapunov: n_positive = 1`, exercised by
    # test_known_values.py, plus the estimator's own tests in
    # test_dde_lyapunov.py).  The other five carry a chaos tag that nothing
    # currently measures — a REAL coverage gap, recorded here so it is visible
    # rather than absent.  Closing it means a slow-tier DDE claim predicate on
    # DelaySystem.lyapunov_spectrum; it is out of this module's flow-and-map
    # scope, not out of scope for the catalogue.
    "MackeyGlass": "DDE — history-space estimator; n_positive pinned by test_known_values.py",
    "IkedaDelay": "DDE — history-space estimator; chaos tag NOT measured anywhere (gap)",
    "PiecewiseCircuit": "DDE — history-space estimator; chaos tag NOT measured anywhere (gap)",
    "ScrollDelay": "DDE — history-space estimator; chaos tag NOT measured anywhere (gap)",
    "SprottDelay": "DDE — history-space estimator; chaos tag NOT measured anywhere (gap)",
    "VossDelay": "DDE — history-space estimator; chaos tag NOT measured anywhere (gap)",
    # SDE: a stochastic process has no deterministic tangent flow, so there is
    # no exponent to measure at all.  Its 'mean-reverting' claim is checked
    # instead by the analytic-moment tests in tests/test_sde_coverage.py
    # (test_ou_component_sample_mean_and_variance_match_analytic_law).
    "OrnsteinUhlenbeck": "SDE — no tangent flow; OU moments checked in test_sde_coverage.py",
}


def test_the_gate_names_the_claims_it_does_not_measure() -> None:
    """Every claim-carrying system is either measured or **named** as unmeasured.

    The exponent sweep silently drops a system three ways — a non-flow family,
    a :data:`LYAPUNOV_EXCLUDE` entry, or a claim the classifier does not reach —
    and a silent drop is indistinguishable from a passing test.  (This is not
    hypothetical: ``Chua``, ``BelousovZhabotinsky`` and ``MacArthur`` each
    carried a chaos tag and sat in ``LYAPUNOV_EXCLUDE``, so nothing measured
    them; on measurement two passed in under 10 s and the third turned out to
    *fail* its claim.)  So the difference between "claims something" and "is
    checked" is asserted to equal an explicit, reasoned roster.
    """
    editorial = _load_editorial()
    claimed = {
        e.name for e in registry.all_systems() if behaviour_claim(e.name, editorial) != CLAIM_NONE
    }
    measured = {p.values[0].name for p in _CHAOTIC_PARAMS} | {
        p.values[0].name for p in _REGULAR_PARAMS
    }
    assert sorted(claimed - measured) == sorted(UNMEASURED_CLAIMS), (
        "the set of systems that make an exponent claim but are not measured here "
        "changed.\n"
        f"  now:      {sorted(claimed - measured)}\n"
        f"  expected: {sorted(UNMEASURED_CLAIMS)}\n"
        "A system that claims chaos and is not measured is exactly the silent skip "
        "this module exists to end: either measure it (drop the LYAPUNOV_EXCLUDE "
        "entry) or add it to UNMEASURED_CLAIMS with the reason and the test that "
        "does cover it."
    )


@pytest.mark.slow
def test_lyapunov_exclusions_are_all_live() -> None:
    """No :data:`LYAPUNOV_EXCLUDE` entry is an affordable system in disguise.

    The exclusions are a *cost* argument, so the cost is measured rather than
    asserted: an entry whose leading exponent computes in a few seconds is not
    too expensive to check and must go back into the sweep.  Every current entry
    is a method-of-lines field of dimension 32-4608, so the guard is run as a
    dimension floor — computing their spectra to prove they are slow would cost
    exactly what the exclusion exists to avoid.  ``KuramotoSivashinsky`` sits
    exactly on the floor at dim 32 (a 32-vector state carries a 32x32 tangent
    bundle, i.e. 1056 coupled equations), so lowering its grid would correctly
    fail this and force the exclusion to be re-argued.
    """
    for name, reason in LYAPUNOV_EXCLUDE.items():
        # `entry.dim` is None for a variable-dimension system (the fields declare
        # their grid through `_structural_params`), so ask the instance.
        dim = registry.get(name).cls().dim
        assert dim >= 32, (
            f"{name} is excluded from the exponent sweep ({reason}) but has "
            f"dimension {dim} — that is not a cost argument. Measure its "
            f"leading exponent and either drop the exclusion or record the "
            f"discrepancy in CHAOS_CLAIM_UNRESOLVED / REGULAR_CLAIM_UNRESOLVED."
        )


def test_every_recorded_discrepancy_names_a_live_system() -> None:
    """The discrepancy tables refer to systems the gate actually judges.

    Cheap bookkeeping that stops an entry outliving its system (or its claim):
    a name that is no longer in the catalogue, no longer carries the matching
    behaviour tag, or has been moved into ``LYAPUNOV_EXCLUDE`` would otherwise
    sit there forever suggesting a finding that is not being checked.
    """
    editorial = _load_editorial()
    judged = {
        CLAIM_CHAOTIC: {p.values[0].name for p in _CHAOTIC_PARAMS},
        CLAIM_REGULAR: {p.values[0].name for p in _REGULAR_PARAMS},
    }
    for claim, table in (
        (CLAIM_CHAOTIC, CHAOS_CLAIM_UNRESOLVED),
        (CLAIM_REGULAR, REGULAR_CLAIM_UNRESOLVED),
    ):
        stale = sorted(set(table) - judged[claim])
        assert not stale, (
            f"recorded {claim} discrepancies name systems the gate no longer judges: "
            f"{stale} (removed from the catalogue, re-tagged, or excluded). Re-check "
            f"and drop the entry — or the finding stops being checked."
        )
        for name in table:
            assert behaviour_claim(name, editorial) == claim, name


# --------------------------------------------------------------------------- #
# Cost — a gate nobody can afford to run is not a gate
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_gate_is_affordable() -> None:
    """The whole fast-tier reference-orbit sweep costs seconds, not minutes.

    A gate nobody can afford to run is not a gate, so the budget is asserted
    rather than assumed.  Measured: 5.1 s of integration for all 177 systems
    (the two method-of-lines fields, GrayScott and SwiftHohenberg, are ~60 % of
    it).  The ceiling is ~6x that — enough headroom for a loaded or slower
    machine, tight enough to catch a regression in kind, such as a new system
    with a runaway window or a per-system integration sneaking into the loop.
    """
    _ORBIT_CACHE.clear()
    _STATS_CACHE.clear()
    started = time.perf_counter()
    for entry in registry.all_systems():
        reference_orbit(entry)
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0, (
        f"the reference-orbit sweep over {len(list(registry.all_systems()))} systems "
        f"took {elapsed:.1f}s; it is budgeted at ~5 s"
    )
