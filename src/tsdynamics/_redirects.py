"""The v6 redirect tables — pure data, sorted by key, **append-only**.

A curated namespace hides several hundred reachable names from autocomplete, so
the exception a wrong guess raises is the *only* feedback a user gets.  These
tables are what turns that exception into a migration guide: one row per name
that moved or left, carrying the line to type instead.

Three tables, three kinds of hit
--------------------------------
:data:`RENAMED_IN_V6`
    The capability is still here under **another spelling**.  The answer is a
    redirect, not an explanation.
:data:`REMOVED_IN_V6`
    The capability **left the library** (the v6 scope surgery moved the generic
    time-series layer to a companion package).  The answer has to explain the
    scope change, or a v5 user reads "no attribute" and assumes a broken install.
:data:`SCOPE_SURGERY_REMEDY`
    The four runnable lines every :data:`REMOVED_IN_V6` row closes with — the
    phase-space methods that *stayed*.

Why this module exists at all
-----------------------------
It is **data only**, imports nothing, and is sorted by key so that several
builders appending rows in the same release merge cleanly.  The machinery that
reads it — the ordered ``__getattr__`` ladder — lives in
:mod:`tsdynamics.__init__`.  Keep the split: a row is a fact about one name, and
adding one must never require touching a function.

**Append your own rows here.**  If your change removes or renames a public name,
its row belongs in this file, in the same commit, sorted into place.

How a row is rendered
---------------------
An exact hit in either table raises :class:`tsdynamics.errors.MovedInV6`, which
is an :class:`ImportError` — *not* an ``AttributeError`` — because a module
``__getattr__`` that raises ``AttributeError`` has its message **discarded** by
``from tsdynamics import X``, the spelling the corpus uses 111 times:

.. code-block:: text

    __getattr__ raises AttributeError -> ImportError: cannot import name 'X'  (TEXT LOST)
    __getattr__ raises ImportError    -> ImportError: <the custom text>       (TEXT KEPT)

A *guess* (a near miss, an unrecognisable name) stays an ``AttributeError``, so
``hasattr(ts, "anything")`` keeps answering ``False`` for every name in the
universe except the enumerated dead ones.
"""

from __future__ import annotations

__all__ = [
    "OUT_OF_SCOPE_SEARCH_TERMS",
    "REMOVED_IN_V6",
    "RENAMED_IN_V6",
    "SCOPE_SURGERY_REMEDY",
]


#: Names the v6 namespace curation removed **in favour of another spelling that
#: still exists**, mapped to ``(line to type, why it changed)``.
#:
#: Distinct from :data:`REMOVED_IN_V6`, where the capability itself left: here
#: nothing was lost, so the answer is a redirect.
#:
#: A name that merely *moved* (``ts.correlation_dimension`` →
#: ``ts.analysis.correlation_dimension``) does **not** belong here — the ladder
#: finds those by looking the name up in each public home's ``__all__``, which
#: cannot go stale the way a hand-written table does.  This table is only for
#: names whose *spelling* changed.
RENAMED_IN_V6: dict[str, tuple[str, str]] = {
    "EnsembleSystem": (
        "ts.derived.Ensemble",
        "the class is called Ensemble now — the -System suffix said nothing the "
        "other four derived wrappers did not also say, and sys.ensemble(states) "
        "is the verb that builds one",
    ),
    "PlotSpec": (
        "ts.viz.Plot",
        "it is the type ts.plot hands back, and nothing about it is a spec any "
        "more — same class, renamed, nothing wrapped",
    ),
    "basins_of_attraction": (
        "ts.analysis.basins(system, region)",
        "basins is the noun a user types; the collision with the analysis.basins "
        "subpackage that used to block the short name is gone, because the "
        "capability subpackages are unbound from ts.analysis now",
    ),
    "bifurcation_diagram": (
        'ts.analysis.orbit_diagram(system, "r", values)',
        "it was a second name for orbit_diagram, and a shared implementation can "
        "name only one of its spellings in an error — so half of all callers were "
        "sent to look up a function they had never typed",
    ),
    "find_attractors": (
        "ts.analysis.attractors(system, region)",
        "the verb was carrying no information: every analysis finds something, and "
        "what this one returns is the attractors",
    ),
    "max_lyapunov": (
        "ts.analysis.lyapunov_spectrum(system, k=1)",
        "two doors onto one question answered it with two numbers — on Henon at "
        "one nominal horizon max_lyapunov said 0.4233 and lyapunov_spectrum said "
        "0.4160 — so the better half (the burn-in, and the Jacobian-free "
        "two-trajectory machine) moved into lyapunov_spectrum and the second "
        "door closed",
    ),
    "periodic_orbit": (
        "ts.analysis.periodic_orbits(system, period_guess, ic=x0)",
        "one verb, one return type: a flow's limit cycle now comes back as an "
        "OrbitSet of one, exactly like a map's orbits — the near-miss suggestion "
        "would have sent you to the plural without saying the return type moved",
    ),
}


#: Names the v6 **scope surgery** removed outright, mapped to the capability
#: group that left.  TSDynamics is a dynamical-systems library: phase-space
#: methods stayed, generic series statistics went to a companion package.
REMOVED_IN_V6: dict[str, str] = {
    "aaft": "the surrogate-data tests",
    "approximate_entropy": "the entropy estimators",
    "butterworth": "the signal-transform toolbox",
    "detrend": "the signal-transform toolbox",
    "dispersion_entropy": "the entropy estimators",
    "entropy": "the entropy estimators",
    "extract_features": "the signal-transform toolbox",
    "ft_surrogate": "the surrogate-data tests",
    "hjorth": "the signal-transform toolbox",
    "iaaft": "the surrogate-data tests",
    "lempel_ziv": "the entropy estimators",
    "lz76_complexity": "the entropy estimators",
    "multiscale_entropy": "the entropy estimators",
    "nonlinear_prediction_error": "the surrogate-data tests",
    "permutation_entropy": "the entropy estimators",
    "power_spectrum": "the signal-transform toolbox",
    "sample_entropy": "the entropy estimators",
    "shuffle_surrogate": "the surrogate-data tests",
    "spectrogram": "the signal-transform toolbox",
    "surrogate": "the surrogate-data tests",
    "surrogate_test": "the surrogate-data tests",
    "surrogates": "the surrogate-data tests",
    "time_reversal_asymmetry": "the surrogate-data tests",
    "transforms": "the signal-transform toolbox",
}


#: The free-text words that mean "you are looking for something the v6 scope
#: surgery removed".  :func:`tsdynamics.analysis.find` consults this so a search
#: for a deleted capability is answered with the scope change rather than with
#: "nothing matches" — which reads as "this library cannot do that", the wrong
#: conclusion, and is what sent a reader off to reimplement an FT surrogate test
#: by hand.
OUT_OF_SCOPE_SEARCH_TERMS: dict[str, str] = {
    "aaft": "the surrogate-data tests",
    "bandpass": "the signal-transform toolbox",
    "butterworth": "the signal-transform toolbox",
    "detrend": "the signal-transform toolbox",
    "entropy": "the entropy estimators",
    "filter": "the signal-transform toolbox",
    "iaaft": "the surrogate-data tests",
    "psd": "the signal-transform toolbox",
    "spectrogram": "the signal-transform toolbox",
    "surrogate": "the surrogate-data tests",
    "surrogates": "the surrogate-data tests",
}


#: The runnable lines a :data:`REMOVED_IN_V6` answer closes with: what stayed.
#:
#: These used to be the module *paths* ``ts.analysis.recurrence`` /
#: ``.embedding`` / ``.lyapunov``.  Two things were wrong with that: a module
#: path is not a line you can type and run, and since v6 unbound the capability
#: subpackages from ``ts.analysis`` those three paths do not even resolve.  Every
#: line below is a call.
SCOPE_SURGERY_REMEDY: tuple[str, ...] = (
    "ts.analysis.rqa(traj)                      # recurrence quantification",
    "ts.analysis.embed(data, dimension, delay)  # data -> phase space",
    "ts.analysis.lyapunov_from_data(traj)       # an exponent from a series",
    'ts.analysis.find("recurrence")             # everything that survived',
)
