"""The :class:`AnalysisResult` base class.

Split out of ``analysis/_result.py``; see that module's docstring (now the
re-exporting facade) for the full result-layer contract.  The repr/JSON helpers
live in :mod:`tsdynamics.analysis._result_json` and the ``.plot`` seam in
:mod:`tsdynamics.analysis._result_viz`; this module holds only the base class.

The repr **is** the answer (v6)
-------------------------------
Before v6 the readable text lived in ``summary()``, which nothing advertised,
while ``__repr__`` — the thing a REPL and a notebook actually show — gave a
constructor-shaped one-liner.  v6 deletes ``summary()`` and makes the repr what
it printed.  Every result renders as::

    <Name>  <THE ANSWER>   <verdict>   (<subject>)
        <up to four supporting lines>
        [0] <item>
        ...
        ... [N total]

built from four subclass hooks, none of which a subclass is obliged to override:

``_answer()``
    The measurement, in the reader's units.  Defaults to the ``_repr_fields``
    rendering, so a result that declares nothing still says what it holds.
``_interpretation()``
    The verdict — *chaotic*, *deterministic*, *not applicable*.  A verdict must
    be supported by the data (contract §4.2 rule 9); return ``None`` to stay
    silent rather than print a measured-looking zero.
``_context()``
    The trailing parenthetical: the originating system by default, or the
    settings that make the number meaningful (``(kantz, m=5, τ=40)``).
``_details()`` / ``_item_lines()``
    Indented supporting lines and, for a collection, the truncated item list.

Because every renderer (``__str__``, ``_repr_html_``, ``__format__``) is derived
from those hooks, a subclass gets the console, the notebook and the f-string for
one override instead of three.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from tsdynamics.analysis._result_json import _fmt, _jsonify, _row_for
from tsdynamics.analysis._result_viz import VisualizationNotInstalled, _PlotAccessor
from tsdynamics.utils.plot_namespace import plot_seam_error

if TYPE_CHECKING:
    from tsdynamics.viz.spec import PlotSpec

#: Separator between the headline's slots (name/answer use two spaces; the
#: verdict and the subject are set further off so the eye finds them).
_GAP = "   "

#: Indent of every supporting / item line under the headline.
_INDENT = "    "

#: A headline plus at most this many supporting lines.  More than four and the
#: repr stops being a readout and becomes a dump.
_MAX_DETAILS = 4

#: Items shown before a collection's list is truncated with ``... [N total]``.
_MAX_ITEMS = 10

#: Format presentation types that mean "render me as a NUMBER".  ``f"{r:.3f}"``
#: on a result that is not one used to fall through to ``format(str(self), spec)``
#: and raise ``ValueError: Unknown format code 'f' for object of type 'str'`` —
#: naming ``str``, a type the caller never typed.
_NUMERIC_FORMAT_CODES = frozenset("bcdeEfFgGnoxX%")

#: The closed vocabulary of **verdict** properties — one adjective-named boolean
#: per classifying result (contract §4.2 rule 10).  Before v6 the same concept
#: had six spellings (``chaotic`` / ``is_chaotic(threshold=)`` / ``is_wada`` /
#: ``fractal_boundary`` / ``trusted`` / ``stable``), so a caller had to learn one
#: per analysis.  Each returns ``None`` — never ``False`` — when the test did not
#: apply.  Used by :meth:`AnalysisResult.__bool__` to name the right thing to
#: test, and by the gate that walks every subclass.
_VERDICT_NAMES = frozenset(
    {
        "applicable",
        "chaotic",
        "deterministic",
        "final_state_sensitive",
        "fractal_boundary",
        "saturated",
        "stable",
        "trusted",
        "wada",
    }
)


def _unknown_result_attribute(result: Any, name: str) -> AttributeError:
    """Build the ``AttributeError`` for a wrong guess on an analysis result.

    Names the result class (not ``numpy.ndarray``), suggests the nearest thing
    the result really carries, and points at the two doors that always work.
    """
    import difflib

    cls = type(result)
    if name == "to_plot_spec":
        return plot_seam_error(cls.__name__, "result")
    carried = sorted(
        {n for n in dir(cls) if not n.startswith("_") and not callable(getattr(cls, n, None))}
        | {n for n in getattr(result, "_repr_fields", ()) if not n.startswith("_")}
        # Names a result serves through its OWN ``__getattr__`` (a windowed RQA's
        # nine measures) are invisible to ``dir``, so they are declared.
        | {n for n in getattr(result, "_extra_attribute_names", ()) if not n.startswith("_")}
    )
    close = difflib.get_close_matches(name, carried, n=2, cutoff=0.55)
    lines = [f"result.{c}" for c in close]
    lines += ["result.to_dict(full=True)   # everything it knows", "print(result)"]
    return AttributeError(
        f"{cls.__name__!r} object has no attribute {name!r}."
        + ("  Did you mean:" if close else "")
        + "".join(f"\n    {line}" for line in lines)
    )


@dataclass(frozen=True)
class AnalysisResult:
    """Base class for every analysis result object.

    See the module docstring for the full contract.  Subclasses re-apply
    ``@dataclass(frozen=True)``, declare their own fields, list the ones to show
    in ``__repr__``/:meth:`summary` via the ``_repr_fields`` class attribute, and
    optionally override :meth:`_interpretation`.

    Attributes
    ----------
    meta : Mapping
        Provenance for the computation: the originating system, its parameters,
        the library version, and the run settings.  Keyword-only; defaults to an
        empty dict.  Build it with :meth:`build_meta`.
    """

    #: Names of the fields shown in ``__repr__`` / :meth:`summary` / the HTML
    #: table.  When empty the dataclass fields are used (skipping ``meta`` and
    #: any declared with ``field(repr=False)``).
    _repr_fields: ClassVar[tuple[str, ...]] = ()

    #: ``meta`` is provenance, not identity: ``compare=False`` keeps it out of the
    #: generated ``__eq__`` / ``__hash__`` (a dict is unhashable, and two otherwise
    #: identical results should compare equal even with differing run provenance),
    #: while it stays in :func:`dataclasses.fields` so :meth:`to_dict` still emits it.
    meta: Mapping[str, Any] = field(default_factory=dict, kw_only=True, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Keep the inherited ``__repr__`` on every subclass.

        ``@dataclass`` regenerates ``__repr__`` for each subclass (shadowing the
        inherited one), which would drop ``_repr_fields`` formatting.  This hook
        runs *before* the decorator, so claiming the MRO-resolved ``__repr__`` in
        the subclass's own namespace makes ``dataclass`` leave it alone.  A
        subclass keeps the nearest ancestor's repr — the base ``_repr_fields``
        one for the common case, or a custom repr a parent defined (so a
        grandchild inherits it too) — unless it writes its own.
        """
        super().__init_subclass__(**kwargs)
        if "__repr__" not in cls.__dict__:
            inherited_repr = cls.__repr__
            cls.__repr__ = inherited_repr  # type: ignore[method-assign]

    def __getattr__(self, name: str) -> Any:
        """Teach the one retired plot name; leave every other miss verbatim.

        Only ``to_plot_spec`` is intercepted (v6 moved the seam to the dunder
        ``__plot_spec__``).  Anything else re-raises the message CPython would
        have produced, so this hook adds no new behaviour to a wrong guess —
        :class:`~tsdynamics.analysis._result_array.ArrayResult` keeps its own,
        richer ``__getattr__``, which routes the same name through
        :func:`_unknown_result_attribute`.
        """
        if name == "to_plot_spec":
            raise plot_seam_error(type(self).__name__, "result")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    # -- provenance -------------------------------------------------------

    @staticmethod
    def build_meta(system: Any = None, **extra: Any) -> dict[str, Any]:
        """Build a provenance dict for a result, from a system plus run settings.

        Delegates to ``system._provenance(**extra)`` when ``system`` exposes it
        (every :class:`~tsdynamics.families.base.SystemBase`), so a result's
        ``meta`` matches the provenance attached to trajectories.  Falls back to
        a minimal ``{"system": ..., **extra}`` for plain objects or ``None``.

        Parameters
        ----------
        system : object, optional
            The system the analysis ran on.
        **extra
            Additional run settings to record (e.g. ``transient=50``).

        Returns
        -------
        dict
        """
        provenance = getattr(system, "_provenance", None)
        if callable(provenance):
            return dict(provenance(**extra))
        if system is None:
            return dict(extra)
        name = getattr(type(system), "__name__", str(system))
        return {"system": name, **extra}

    # -- repr / summary ---------------------------------------------------

    def _display_fields(self) -> tuple[str, ...]:
        """Resolve the field names to display (``_repr_fields`` or introspection)."""
        if self._repr_fields:
            return self._repr_fields
        return tuple(f.name for f in fields(self) if f.repr and f.name != "meta")

    #: Result classes whose *name* carries no domain meaning.  A named result
    #: (``LyapunovSpectrum``, ``RQAResult``) reprs as itself; these generic
    #: wrappers repr as ``ScalarResult(max_lyapunov, value=0.423267)`` instead of
    #: an anonymous ``ScalarResult(value=0.423267)``, because in a console the
    #: repr is the only thing that says *what was measured*.
    _ANONYMOUS_RESULT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"ScalarResult", "CountResult", "ArrayResult", "CollectionResult", "ScalingResult"}
    )

    # -- the four repr hooks ----------------------------------------------

    def _result_name(self) -> str:
        """Name the headline opens with.

        A named result reprs as itself (``LyapunovSpectrum``, ``RQAResult``); an
        anonymous wrapper reprs under the *analysis* that produced it, because
        in a console the repr is the only thing that says what was measured.
        """
        name = type(self).__name__
        if name in self._ANONYMOUS_RESULT_TYPES:
            analysis = self.meta.get("analysis") if self.meta else None
            if analysis:
                return str(analysis)
        return name

    def _is_anonymous(self) -> bool:
        """Whether the headline name came from ``meta`` rather than the class."""
        return type(self).__name__ in self._ANONYMOUS_RESULT_TYPES and bool(
            self.meta.get("analysis") if self.meta else None
        )

    def _subject_family(self) -> str | None:
        """Return the producing system's family (``ode`` / ``map`` / …), if knowable.

        Read off the registry by the name ``meta["system"]`` records, falling
        back to the horizon keyword the analysis stored (``final_time`` ⇒ a flow,
        ``n`` ⇒ a map).  ``None`` when the result came from measured data with no
        system behind it.  Used wherever the *units* of an answer depend on
        whether time is continuous — a Lyapunov exponent is per unit time for a
        flow and per iteration for a map, and printing the wrong one is a
        wrong answer, not a cosmetic slip.
        """
        if not self.meta:
            return None
        name = self.meta.get("system")
        if name:
            from tsdynamics import registry

            try:
                entry = registry.get(str(name))
            except (KeyError, LookupError):
                entry = None
            if entry is not None:
                return str(entry.family)
        if self.meta.get("final_time") is not None:
            return "ode"
        if self.meta.get("n") is not None:
            return "map"
        return None

    def _answer(self) -> str:
        """Return **the answer**: what was measured, in the reader's units.

        The base renders the declared display fields (``_repr_fields``, else the
        dataclass fields) as ``name = value`` joined by ``·``, so a result that
        overrides nothing still says what it holds.
        """
        parts = []
        for name in self._display_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            parts.append(f"{name} = {_fmt(value)}")
        return " · ".join(parts)

    def _interpretation(self) -> str | None:
        """Return the one-word/one-clause verdict, or ``None`` to stay silent.

        Overridden by subclasses (``chaotic``, ``deterministic``, ``fractal
        boundary``).  A verdict must be **supported by the data**: when the test
        does not apply, return ``None`` rather than a measured-looking negative.
        """
        return None

    def _context(self) -> str | None:
        """Return the trailing parenthetical, or ``None`` for none.

        By default the originating system, which is what makes an answer
        attributable.  A result whose meaning depends on its settings rather
        than on a system (an embedding's ``(m=3, τ=9 samples)``, a recurrence
        matrix's ``(euclidean, theiler=0)``) overrides this with those instead.
        """
        return self._system_label()

    def _details(self) -> tuple[str, ...]:
        """Return the supporting lines under the headline (at most four)."""
        return ()

    def _item_lines(self) -> tuple[str, ...]:
        """Return the item lines of a collection-like result (already truncated)."""
        return ()

    def _as_item(self) -> str:
        """Return this result's one-line form *inside another result's list*.

        A collection lists its members, and a member's headline repeats the
        subject and the class name on every row — noise, when the header already
        said both.  A result that is commonly collected (a fixed point, an
        attractor, a periodic orbit) overrides this with the compact form; the
        default is the headline, which is right for anything else.

        Kept separate from :meth:`__str__` deliberately: ``str(result)`` is the
        headline for *every* result (contract §4.2 rule 12), so overloading it
        for the list form would make ``print(fixed_point)`` drop the system it
        belongs to.
        """
        return self.headline()

    def _system_label(self) -> str | None:
        """Return the originating system's name from ``meta``, if recorded."""
        system = self.meta.get("system") if self.meta else None
        return str(system) if system else None

    # -- the renderers, all derived from the hooks above ------------------

    def headline(self) -> str:
        """Return the single line that carries the answer.

        ``<Name>  <answer>   <verdict>   (<subject>)``, with the empty slots
        omitted.  This is the repr's first line and the whole of ``str(self)``.

        Returns
        -------
        str
        """
        line = self._result_name()
        answer = self._answer()
        if answer:
            # A named result is a title followed by its readout, so it gets the
            # wider gap; an anonymous one reads as a single sentence
            # (``max_lyapunov = 0.42 per iteration``) and gets one space.
            line += (" " if self._is_anonymous() else "  ") + answer
        verdict = self._interpretation()
        if verdict:
            line += _GAP + verdict
        context = self._context()
        if context:
            line += _GAP + f"({context})"
        return line

    def __repr__(self) -> str:  # noqa: D105
        lines = [self.headline()]
        lines += [_INDENT + text for text in self._details()[:_MAX_DETAILS]]
        lines += [_INDENT + text for text in self._item_lines()]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the headline — the answer without the supporting lines."""
        return self.headline()

    def _as_number(self) -> float | None:
        """Return the number this result stands for, or ``None`` if it is not one.

        Reads whichever numeric conversion the subclass declares — ``__float__``
        for a measured quantity, ``__int__`` for a count or a dimension — so a
        result that *is* a number never has to also spell out ``__format__``.
        """
        for name in ("__float__", "__int__"):
            convert = getattr(type(self), name, None)
            if convert is None:
                continue
            try:
                return float(convert(self))
            except (TypeError, ValueError):
                return None
        return None

    def _numeric_field_names(self) -> tuple[str, ...]:
        """Return the fields that hold numbers or numeric arrays, in declared order.

        What :meth:`__array__` names when it refuses, so the message says *what
        to ask for* instead of leaving the caller to guess.
        """
        names: list[str] = []
        for f in fields(self):
            if f.name == "meta":
                continue
            try:
                value = getattr(self, f.name)
            except AttributeError:  # pragma: no cover - defensive
                continue
            numeric = isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)) or (
                isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
            )
            if numeric:
                names.append(f.name)
        return tuple(names)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Return this result as a real numeric array — or refuse, by name.

        ``np.asarray(result)`` used to hand back a **0-d object array** for 17 of
        the 32 result classes: it did not raise, it produced
        ``array(RQAResult  DET = 0.794 …, dtype=object)``, which plots as nothing
        and arithmetics into a ``TypeError`` far from the call site.

        One rule decides (contract §4.2 rule 5):

        - a result that **is** a number (it declares ``__float__`` / ``__int__``)
          arrays as that number, 0-d, exactly like the scalar it replaced;
        - a result that **is** a collection or a field overrides this with its
          natural array (a spectrum's exponents, a set's representatives, a basin
          image's labels);
        - anything else **raises** ``TypeError`` naming the numeric fields it
          does carry, so the message is the next thing to type.

        Raises
        ------
        TypeError
            If the result is neither a number nor an array of them.
        """
        number = self._as_number()
        if number is None:
            carried = self._numeric_field_names()
            suggestion = (
                f"e.g. np.asarray(result.{carried[0]})"
                if carried
                else "use result.to_dict() for everything it knows"
            )
            raise TypeError(
                f"{type(self).__name__} is not an array of numbers."
                + (f"  Its numeric parts are: {', '.join(carried)}." if carried else "")
                + f"\n    {suggestion}"
                + "\n    result.to_dict()   # everything it knows"
            )
        arr = np.asarray(number)
        if dtype is not None:
            arr = arr.astype(dtype, copy=bool(copy))
        elif copy:
            arr = arr.copy()
        return arr

    def __bool__(self) -> bool:
        """Refuse the coin flip — a measurement is not a flag (contract §4.2 rule 7).

        ``bool(result)`` was ``True`` for 31 of the 32 classes, so
        ``if ts.analysis.wada_property(...):`` fired whether or not the boundary
        was Wada.  A **sized** result keeps Python's own convention (non-empty is
        true), a result that *is* a number keeps the number's, and everything
        else raises, naming the verdict to test instead.

        Raises
        ------
        TypeError
            If the result is neither sized nor a number.
        """
        sized = getattr(type(self), "__len__", None)
        if sized is not None:
            return bool(sized(self) > 0)
        verdicts = self._verdict_property_names()
        if verdicts:
            named = f"`if result.{verdicts[0]}:` (the verdict)"
            if "applicable" in verdicts:
                named += ", or `if result.applicable:` (did the test apply)"
        else:
            carried = self._numeric_field_names()[:3]
            named = (
                "the comparison you mean, e.g. `if result." + carried[0] + " > 0:`"
                if carried
                else "`if result.to_dict():`"
            )
        raise TypeError(
            f"the truth value of a {type(self).__name__} is ambiguous — it is a "
            f"measurement, not a flag.\n    Write {named}."
        )

    def _verdict_property_names(self) -> tuple[str, ...]:
        """Return this class's verdict properties, from the closed vocabulary."""
        return tuple(
            sorted(
                n
                for n in _VERDICT_NAMES
                if isinstance(getattr(type(self), n, None), property)
                or n in {f.name for f in fields(self)}
            )
        )

    @property
    def verdict(self) -> str | None:
        """The clause the repr prints as its verdict, or ``None`` when it has none.

        The one spelling for "what did this measurement conclude?", in the exact
        words the headline uses, so a caller never has to parse the repr to get
        it.  :meth:`to_dict` emits it under ``"verdict"`` at ``full=True``.

        Returns
        -------
        str or None
        """
        return self._interpretation()

    def __format__(self, spec: str) -> str:
        """Format the underlying number when a format spec is given.

        ``f"{result:.3f}"`` used to raise ``TypeError`` on every numeric result,
        which made a result a *worse* drop-in for the number it replaced than the
        bare float it wrapped.  An empty spec defers to :meth:`__str__` (so
        ``f"{result}"`` prints the headline); a non-empty one formats the number
        when the result is one, and otherwise formats the headline text (so
        ``f"{result:>40}"`` still aligns a non-numeric result).

        A **numeric** presentation code on a non-numeric result raises
        ``TypeError`` naming this class: it used to fall through to
        ``format(str(self), spec)`` and surface ``ValueError: Unknown format code
        'f' for object of type 'str'``, which names ``str`` — a type the caller
        never typed.
        """
        if not spec:
            return str(self)
        number = self._as_number()
        if number is None:
            if spec[-1] in _NUMERIC_FORMAT_CODES:
                carried = self._numeric_field_names()
                pick = f"result.{carried[0]}" if carried else "one of its fields"
                raise TypeError(
                    f'f"{{result:{spec}}}" needs a number; {type(self).__name__} is not one.'
                    f'\n    Format {pick}, or use f"{{result}}" for the headline.'
                )
            return format(str(self), spec)
        return format(number, spec)

    def _repr_html_(self) -> str:
        """Return the repr, verbatim, for Jupyter / IPython.

        The notebook and the console show the **same** text.  A bespoke HTML
        table was a second rendering of the same result that drifted from the
        console one and had to be re-derived per subclass; ``<pre>`` of the repr
        cannot drift.
        """
        return f"<pre style='white-space:pre;margin:0'>{html.escape(repr(self))}</pre>"

    # -- export -----------------------------------------------------------

    def _derived(self) -> dict[str, Any]:
        """Return the named quantities the repr reports that are **not** fields.

        A result's headline usually reports a derived property — a spectrum's
        ``kaplan_yorke``, a recurrence matrix's ``recurrence_rate``, a Wada
        test's ``W`` — which :func:`dataclasses.fields` cannot see, so
        ``to_dict()`` could not export the very number the repr shows.
        ``to_dict(full=True)`` adds these.  Subclasses override; the base has
        none.
        """
        return {}

    def _full_extras(self) -> dict[str, Any]:
        """Return everything ``to_dict(full=True)`` adds — derived + verdict + unit.

        The contract *"``full`` emits what the repr reports"* used to be a
        per-subclass promise, and 14 of the 32 added **nothing** — including
        every result whose repr shows a verdict.  Building it here makes it true
        by construction for every subclass, including the four that override
        :meth:`to_dict` (they call this).
        """
        extras: dict[str, Any] = dict(self._derived())
        extras.setdefault("verdict", self.verdict)
        unit = getattr(self, "_unit", None)
        if callable(unit):
            extras.setdefault("unit", unit())
        return extras

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the result (arrays become lists).

        Uses only the standard library.  Every dataclass field is included
        (``meta`` too), recursively coerced to JSON-serializable types.

        Parameters
        ----------
        full : bool, default False
            Also emit the **derived** quantities the repr reports but that are
            properties rather than fields (see :meth:`_derived`) — a Lyapunov
            spectrum's ``kaplan_yorke``, a recurrence matrix's
            ``recurrence_rate``, a Wada test's ``applicable`` / ``W``.  The
            default view is exactly the declared fields, so ``full`` only ever
            *adds* keys.

        Returns
        -------
        dict
        """
        data = {f.name: _jsonify(getattr(self, f.name)) for f in fields(self)}
        if full:
            data.update({k: _jsonify(v) for k, v in self._full_extras().items()})
        return data

    @staticmethod
    def _require_pandas() -> Any:
        """Import :mod:`pandas` lazily, raising a friendly hint if it is absent."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "to_frame() needs pandas, which TSDynamics does not depend on. "
                "Install it with `pip install pandas`. Use .to_dict() for a "
                "stdlib-only export of the same fields."
            ) from exc
        return pd

    def to_frame(self) -> Any:
        """Return a :class:`pandas.DataFrame` view of the result.

        ``pandas`` is a soft dependency, imported lazily; a missing install
        raises an :class:`ImportError` pointing at ``pip install pandas``.

        The base produces a single-row frame of **the content** — the dataclass
        fields, with short numeric vectors spread into ``name0 name1 …`` columns
        — with ``meta`` carried on ``frame.attrs["meta"]``.  Subclasses that
        carry a natural table (a scaling curve, a wrapped array, a collection)
        override this with one row per member.

        .. versionchanged:: 6.0
           It used to tabulate ``_display_fields`` and keep only the *scalars*,
           so ``to_frame()`` dropped **the answer** on 10 of the 32 classes: a
           fixed point tabulated ``['stable', 'continuous']`` and lost its
           coordinates, while the set it belongs to spread them properly — the
           same information, two answers.  A field that is itself a result, a
           mapping or a grid is dropped rather than parked in a cell.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If :mod:`pandas` is not installed (the message points at
            ``pip install pandas``).
        """
        pd = self._require_pandas()
        row = _row_for(self)
        frame = pd.DataFrame([row]) if row else pd.DataFrame()
        frame.attrs["meta"] = dict(self.meta) if self.meta else {}
        return frame

    # -- visualization seam ----------------------------------------------

    def __plot_spec__(self, kind: str | None = None) -> Any:
        r"""Describe this result as a backend-agnostic :class:`PlotSpec` (generic fallback).

        **This is the internal seam, not a verb you type.**  ``ts.plot``
        classifies a positional argument as a *subject* by asking whether it
        carries ``__plot_spec__`` — one predicate for a system, a trajectory and
        all 32 results.  A result subclass overrides *this* method; the two
        spellings a user types are ``ts.plot(result)`` (hands back the
        :class:`~tsdynamics.viz.spec.Plot`, drawing nothing) and
        ``result.plot()``.

        .. versionchanged:: 6.0
           Was the public ``to_plot_spec``.  ``result.to_plot_spec`` now raises,
           naming both spellings that work.

        Result types with a *natural* figure override this with a bespoke spec
        (a scaling fit, a recurrence image, a phase portrait, …).  This base
        provides the **fallback** every other result inherits, so the ``.plot``
        seam and any registered renderer resolve uniformly across the whole
        analysis layer rather than tripping over a result that never grew a
        bespoke method.

        The fallback inspects the result's display fields (see
        :meth:`_display_fields`) and builds a ``DIAGNOSTIC_CURVE``:

        - the first pair of equal-length 1-D numeric arrays becomes an ``(x, y)``
          ``LINE`` (e.g. a basin-metric's :math:`f(\\varepsilon)` curve);
        - failing that, the first 1-D numeric array becomes a ``LINE`` against
          its index;
        - failing that, the numeric scalar fields become ``MARKERS`` at one point
          per field (e.g. a basin entropy's :math:`S_b` / :math:`S_{bb}`).

        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls a plotting library and the spec itself carries no rendering code.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
            value).  ``None`` uses ``DIAGNOSTIC_CURVE``.

        Returns
        -------
        PlotSpec

        See Also
        --------
        _overlay_on : the ``base=`` overlay convention a result with a host view
            (fixed points over a phase portrait) uses in its own ``__plot_spec__``.

        Raises
        ------
        VisualizationNotInstalled
            If the result carries **no** plottable numeric field (no array and no
            scalar) — there is nothing for a generic fallback to draw, so a
            bespoke ``__plot_spec__`` is required.  Subclasses with such a result
            override this method.
        """
        from . import _plotbuilder as pb

        arrays, scalars = self._plottable_fields()

        layers = []
        xlabel, ylabel = "index", "value"
        if len(arrays) >= 2 and arrays[0][1].size == arrays[1][1].size:
            (xname, x), (yname, y) = arrays[0], arrays[1]
            layers.append(pb.line(x, y, label=yname))
            xlabel, ylabel = xname, yname
        elif arrays:
            yname, y = arrays[0]
            layers.append(pb.line(np.arange(y.size, dtype=float), y, label=yname))
            ylabel = yname
        elif scalars:
            names = [n for n, _ in scalars]
            vals = np.array([v for _, v in scalars], dtype=float)
            return pb.spec(
                kind,
                "diagnostic_curve",
                layers=[pb.markers(np.arange(vals.size, dtype=float), vals)],
                xlabel="field",
                xticks=list(range(len(names))),
                xcategories=names,
                ylabel="value",
                title=type(self).__name__,
                meta=self.meta,
            )
        else:
            raise VisualizationNotInstalled(
                f"{type(self).__name__} carries no plottable numeric field, so the generic "
                "__plot_spec__() fallback has nothing to draw. A result of this kind needs "
                "a bespoke __plot_spec__(); export it with .to_dict() meanwhile."
            )

        return pb.spec(
            kind,
            "diagnostic_curve",
            layers=layers,
            xlabel=xlabel,
            ylabel=ylabel,
            title=type(self).__name__,
            meta=self.meta,
        )

    def overlay_on(
        self, base: PlotSpec, *, kind: str | None = None, on: str | None = None, **build_kw: Any
    ) -> PlotSpec:
        """Overlay this result's figure onto a host ``base`` spec (host drawn first).

        The ``base=`` overlay convention, as a method that does not perturb the
        uniform ``__plot_spec__(self, kind=None)`` signature: build this result's
        spec and append its layers / annotations *after* the host's, so e.g.
        fixed-point markers land over a phase portrait or an attractor scatter
        over a basin image.  The merged ``base`` is mutated and returned.

        **Frame-checked (v6).**  The host and the overlay must be drawings of the
        same space on the same axes (see
        :attr:`~tsdynamics.viz.spec.PlotSpec.resolved_frame`), which is the same
        rule :func:`tsdynamics.viz.plot` applies — before v6 the two doors
        disagreed, and this one accepted, for instance, a recurrence scatter
        spliced onto a time series, producing a spec *labelled* ``time_series``
        containing a recurrence plot.  That now raises.

        Every keyword other than ``on`` is forwarded to :meth:`__plot_spec__`, so
        an overlay that has to be told which plane to draw on (``components=``)
        can be, without each such result re-implementing this method.

        Parameters
        ----------
        base : PlotSpec
            The host spec to draw under this result.
        kind : str, optional
            Forwarded to :meth:`__plot_spec__`.
        on : {"force"}, optional
            ``"force"`` overlays a deliberate frame mismatch with a one-time
            :class:`~tsdynamics.viz.render.caps.VisualizationDegraded` warning
            instead of raising.
        **build_kw
            Forwarded to :meth:`__plot_spec__` (e.g. ``components=`` /
            ``annotate=`` on a fixed-point set).

        Returns
        -------
        PlotSpec
            ``base``, with this result's layers / annotations appended.

        Raises
        ------
        tsdynamics.errors.InvalidParameterError
            If the host and this result draw in incompatible frames and ``on``
            is not ``"force"``.
        """
        return self._overlay_on(self.__plot_spec__(kind=kind, **build_kw), base, on=on)

    @staticmethod
    def _overlay_on(spec: PlotSpec, base: PlotSpec | None, *, on: str | None = None) -> PlotSpec:
        """Overlay ``spec``'s layers/annotations onto ``base`` (host first), or pass through.

        The ``base=`` overlay convention: when a host ``base`` spec is given, its
        layers are drawn first and ``spec``'s layers/annotations are appended, so
        e.g. fixed-point markers land *over* a phase portrait.  Returns ``spec``
        unchanged when ``base`` is ``None``.

        The frame check is :func:`tsdynamics.viz._frames.check_overlay` — the
        *same* function :func:`tsdynamics.viz.plot` calls, so the library has one
        overlay policy rather than two that disagree.  The merge itself stays
        host-first append (rather than ``compose``'s role sort) because this
        method's contract is that it mutates and returns the host you handed it.
        """
        if base is None:
            return spec
        from tsdynamics.viz._frames import check_overlay, force_requested

        check_overlay([base, spec], force=force_requested(on))
        base.layers = list(base.layers) + list(spec.layers)
        base.annotations = list(base.annotations) + list(spec.annotations)
        if base.legend is None and len(base.layers) > 1:
            from tsdynamics.viz.spec import Legend

            base.legend = Legend()
        return base

    def _plottable_fields(
        self,
    ) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, float]]]:
        """Split the display fields into ``(1-D numeric arrays, numeric scalars)``.

        Used by the generic :meth:`__plot_spec__` fallback.  Booleans are treated
        as scalars (``0`` / ``1``); non-numeric and higher-dimensional fields are
        skipped.  ``meta`` is never included (it is provenance, not plot data).
        """
        arrays: list[tuple[str, np.ndarray]] = []
        scalars: list[tuple[str, float]] = []
        for name in self._display_fields():
            try:
                value = getattr(self, name)
            except AttributeError:
                continue
            if isinstance(value, (bool, np.bool_)):
                scalars.append((name, float(value)))
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalars.append((name, float(value)))
                continue
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 1
                and value.size
                and np.issubdtype(value.dtype, np.number)
            ):
                arrays.append((name, np.asarray(value, dtype=float)))
        return arrays, scalars

    @property
    def plot(self) -> _PlotAccessor:
        """The visualization seam — callable, and a namespace of transform names.

        ``result.plot()`` builds this result's own view, styled at the door with
        the same vocabulary every other plotting door takes.
        ``result.plot.<TAB>`` lists the registered transforms that admit **this**
        result, and ``result.plot.scaling_fit()`` is exactly
        ``ts.plot(result, "scaling_fit")`` — the gesture ``traj.plot`` and
        ``system.plot`` already carry, meaning the same thing.

        The in-tree backends seed themselves on first use, so it works out of the
        box when a plotting library is installed; with none it raises
        :class:`VisualizationNotInstalled`.

        .. versionchanged:: 6.0
            The eight kind-forcing methods (``.scaling()`` / ``.phase()`` /
            ``.image()`` / …) are gone: they relabelled this result's spec with
            another kind's name and changed nothing else.  See
            :class:`~tsdynamics.analysis._result_viz._PlotAccessor`.

        Returns
        -------
        _PlotAccessor
            A callable that is also a namespace of transform names.
        """
        return _PlotAccessor(self)
