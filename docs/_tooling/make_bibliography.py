"""Generate the TSDynamics literature bibliography page from the registry.

The **registry is the single source of truth** for the *systems* bibliography:
every built-in system carries a ``reference`` (and, where known, a ``doi``)
ClassVar, and this script walks :func:`tsdynamics.registry.all_systems`,
deduplicates the reference strings, groups them by family, alphabetises each
group by first-author surname, and emits a clickable list — each paper followed
by the catalogue systems that cite it.  A new (or edited) system therefore lands
in the bibliography with **zero manual edits** here: re-run the script.

The *methods* bibliography — the original papers behind the analysis toolkit
(Lyapunov, GALI, RQA, dimensions, entropy, basins, …) — is a curated table
below.  It is **not** registry-derived (methods are code, not catalogue
entries), so it is transcribed here verbatim from the ``## References`` blocks of
the ``docs/analysis/*.md`` pages.  Every entry is a real citation; nothing is
invented, and no DOI is attached unless it is certain.

Run from the repository root::

    .venv/bin/python docs/_tooling/make_bibliography.py

It writes ``docs/references/bibliography.md`` (committed).  A ``--check`` flag
regenerates in memory and diffs against the committed file (non-zero exit on
drift) for CI use.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "docs" / "references" / "bibliography.md"

# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_ROOT / "src"))


def _load_registry():  # noqa: ANN202 — thin lazy import, keeps import cheap
    from tsdynamics import registry

    return registry


# Display order + labels for the system families (mirrors catalog._FAMILY_ORDER).
_FAMILY_ORDER: tuple[str, ...] = ("ode", "dde", "sde", "map")
_FAMILY_LABEL: dict[str, str] = {
    "ode": "Ordinary differential equations",
    "dde": "Delay differential equations",
    "sde": "Stochastic differential equations",
    "map": "Discrete maps",
}


# ---------------------------------------------------------------------------
# Citation parsing (for sort + year extraction only — the reference string
# itself is always rendered verbatim, never reconstructed)
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\((\d{4}[a-z]?)\)")


def _year(ref: str) -> str:
    """Extract the ``(YYYY)`` publication year, or ``""`` if absent."""
    m = _YEAR_RE.search(ref)
    return m.group(1) if m else ""


def _first_surname(ref: str) -> str:
    """First-author surname, for alphabetical ordering.

    The reference format is ``Author(s) (Year), Venue …``; the author list is
    everything before the first parenthesis.  ``et al.`` is stripped, the first
    ``,``/``&``-separated name is taken, and its last whitespace token is the
    surname (so "van Wyk" sorts under W is acceptable — the last token is the
    family name in the vast majority of the catalogue's citations).
    """
    head = ref.split("(", 1)[0].strip()
    head = re.split(r"\bet al\.?", head)[0].strip()
    first = re.split(r"[,&]", head)[0].strip()
    tokens = first.split()
    return (tokens[-1] if tokens else first).lower()


def _sort_key(ref: str) -> tuple[str, str, str]:
    return (_first_surname(ref), _year(ref), ref.lower())


def _doi_url(doi: str) -> str:
    """Build a safe ``https://doi.org/`` URL for a raw DOI string.

    Parentheses are legal in a DOI path and left as-is; the angle brackets
    that appear in a handful of legacy AMS DOIs (e.g. the 1963 Lorenz paper)
    would truncate a Markdown link, so they — and any stray spaces — are
    percent-encoded, which is exactly the form ``doi.org`` resolves.
    """
    return "https://doi.org/" + quote(doi, safe="/()[]:;,.-_")


def _doi_label(doi: str) -> str:
    """Render the visible DOI text, with angle brackets HTML-escaped.

    A raw ``<0130:dnf>`` inside Markdown link text can be mis-parsed as an HTML
    tag; escaping keeps the displayed DOI faithful to the real identifier.
    """
    return doi.replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Systems bibliography (registry-derived)
# ---------------------------------------------------------------------------


@dataclass
class _Paper:
    reference: str
    doi: str | None
    systems: list[str]  # catalogue system names citing this reference


def _collect_systems(registry) -> tuple[dict[str, list[_Paper]], int, int, int]:
    """Build ``{family: [_Paper, …]}`` deduplicated by reference string.

    Returns the grouping plus ``(n_systems, n_papers, n_without_ref)``.
    """
    # family -> reference -> _Paper
    by_family: dict[str, dict[str, _Paper]] = {f: {} for f in _FAMILY_ORDER}
    n_systems = 0
    n_without = 0
    for entry in sorted(registry.all_systems(), key=lambda e: e.name):
        n_systems += 1
        cls = entry.cls
        ref = getattr(cls, "reference", None)
        doi = getattr(cls, "doi", None)
        if not ref:
            n_without += 1
            continue
        bucket = by_family.setdefault(entry.family, {})
        paper = bucket.get(ref)
        if paper is None:
            bucket[ref] = _Paper(reference=ref, doi=doi, systems=[entry.name])
        else:
            paper.systems.append(entry.name)
            # keep a DOI if any citing system carries one
            if paper.doi is None and doi:
                paper.doi = doi

    grouped: dict[str, list[_Paper]] = {}
    n_papers = 0
    for family in _FAMILY_ORDER:
        papers = sorted(by_family.get(family, {}).values(), key=lambda p: _sort_key(p.reference))
        if papers:
            grouped[family] = papers
            n_papers += len(papers)
    return grouped, n_systems, n_papers, n_without


def _render_systems_paper(paper: _Paper) -> str:
    ref = paper.reference
    if paper.doi:
        line = f"- {ref}. [doi:{_doi_label(paper.doi)}]({_doi_url(paper.doi)})"
    else:
        line = f"- {ref}."
    cited = ", ".join(f"`{name}`" for name in sorted(paper.systems))
    line += f'<br><span class="ts-cite-systems">Cited by: {cited}</span>'
    return line


# ---------------------------------------------------------------------------
# Methods bibliography (curated — transcribed from docs/analysis/*.md)
#
# Each entry is (author-surname-sort-key, rendered-markdown-line).  DOIs are
# attached only where certain; the rendered text is the exact citation used on
# the corresponding Analysis page.  DO NOT invent a DOI or a citation here.
# ---------------------------------------------------------------------------

_METHODS: dict[str, list[tuple[str, str]]] = {
    "Lyapunov exponents & tangent-space dynamics": [
        (
            "benettin1976",
            'G. Benettin, L. Galgani & J.-M. Strelcyn, "Kolmogorov entropy and '
            'numerical experiments", *Phys. Rev. A* **14**, 2338 (1976). '
            "[doi:10.1103/PhysRevA.14.2338](https://doi.org/10.1103/PhysRevA.14.2338)",
        ),
        (
            "benettin1980",
            "G. Benettin, L. Galgani, A. Giorgilli & J.-M. Strelcyn, “Lyapunov "
            "characteristic exponents for smooth dynamical systems and for "
            "Hamiltonian systems; a method for computing all of them”, "
            "*Meccanica* **15**, 9 & 21 (1980). "
            "[doi:10.1007/BF02128236](https://doi.org/10.1007/BF02128236)",
        ),
        (
            "kaplan1979",
            "J. L. Kaplan & J. A. Yorke, “Chaotic behavior of "
            "multidimensional difference equations”, in *Functional "
            "Differential Equations and Approximation of Fixed Points*, Lecture "
            "Notes in Mathematics **730**, 204, Springer (1979). "
            "[doi:10.1007/BFb0064319](https://doi.org/10.1007/BFb0064319)",
        ),
        (
            "kantz1994",
            'H. Kantz, "A robust method to estimate the maximal Lyapunov '
            'exponent of a time series", *Phys. Lett. A* **185**, 77 (1994). '
            "[doi:10.1016/0375-9601(94)90991-1](https://doi.org/10.1016/0375-9601(94)90991-1)",
        ),
        (
            "rosenstein1993",
            "M. T. Rosenstein, J. J. Collins & C. J. De Luca, “A practical "
            "method for calculating largest Lyapunov exponents from small data "
            "sets”, *Physica D* **65**, 117 (1993). "
            "[doi:10.1016/0167-2789(93)90009-P](https://doi.org/10.1016/0167-2789(93)90009-P)",
        ),
    ],
    "Chaos indicators": [
        (
            "skokos2007",
            "Ch. Skokos, T. C. Bountis & Ch. Antonopoulos, “Geometrical "
            "properties of local dynamics in Hamiltonian systems: the "
            "Generalized Alignment Index (GALI) method”, *Physica D* "
            "**231**, 30 (2007). "
            "[doi:10.1016/j.physd.2007.04.004](https://doi.org/10.1016/j.physd.2007.04.004)",
        ),
        (
            "gottwald2004",
            "G. A. Gottwald & I. Melbourne, “A new test for chaos in "
            "deterministic systems”, *Proc. R. Soc. Lond. A* **460**, 603 "
            "(2004). "
            "[doi:10.1098/rspa.2003.1183](https://doi.org/10.1098/rspa.2003.1183)",
        ),
        (
            "gottwald2009",
            "G. A. Gottwald & I. Melbourne, “On the implementation of the "
            "0–1 test for chaos”, *SIAM J. Appl. Dyn. Syst.* **8**, "
            "129 (2009). "
            "[doi:10.1137/080718851](https://doi.org/10.1137/080718851)",
        ),
        (
            "hunt2015",
            'B. R. Hunt & E. Ott, "Defining chaos", *Chaos* **25**, 097618 '
            "(2015). [doi:10.1063/1.4922973](https://doi.org/10.1063/1.4922973)",
        ),
    ],
    "Fractal dimensions": [
        (
            "grassberger1983prl",
            "P. Grassberger & I. Procaccia, “Characterization of strange "
            "attractors”, *Phys. Rev. Lett.* **50**, 346 (1983). "
            "[doi:10.1103/PhysRevLett.50.346](https://doi.org/10.1103/PhysRevLett.50.346)",
        ),
        (
            "hentschel1983",
            "H. G. E. Hentschel & I. Procaccia, “The infinite number of "
            "generalized dimensions of fractals and strange attractors”, "
            "*Physica D* **8**, 435 (1983). "
            "[doi:10.1016/0167-2789(83)90235-X](https://doi.org/10.1016/0167-2789(83)90235-X)",
        ),
        (
            "badii1985",
            "R. Badii & A. Politi, “Statistical description of chaotic "
            "attractors: the dimension function”, *J. Stat. Phys.* **40**, "
            "725 (1985). [doi:10.1007/BF01009897](https://doi.org/10.1007/BF01009897)",
        ),
        (
            "grassberger1985",
            "P. Grassberger, “Generalizations of the Hausdorff dimension of "
            "fractal measures”, *Phys. Lett. A* **107**, 101 (1985). "
            "[doi:10.1016/0375-9601(85)90724-8](https://doi.org/10.1016/0375-9601(85)90724-8)",
        ),
        (
            "theiler1990",
            'J. Theiler, "Estimating fractal dimension", *J. Opt. Soc. Am. A* '
            "**7**, 1055 (1990). "
            "[doi:10.1364/JOSAA.7.001055](https://doi.org/10.1364/JOSAA.7.001055)",
        ),
    ],
    "Delay embedding & state-space reconstruction": [
        (
            "takens1981",
            'F. Takens, "Detecting strange attractors in turbulence", in '
            "*Dynamical Systems and Turbulence*, Lecture Notes in Mathematics "
            "**898**, 366, Springer (1981). "
            "[doi:10.1007/BFb0091924](https://doi.org/10.1007/BFb0091924)",
        ),
        (
            "fraser1986",
            "A. M. Fraser & H. L. Swinney, “Independent coordinates for "
            "strange attractors from mutual information”, *Phys. Rev. A* "
            "**33**, 1134 (1986). "
            "[doi:10.1103/PhysRevA.33.1134](https://doi.org/10.1103/PhysRevA.33.1134)",
        ),
        (
            "kennel1992",
            "M. B. Kennel, R. Brown & H. D. I. Abarbanel, “Determining "
            "embedding dimension for phase-space reconstruction using a "
            "geometrical construction”, *Phys. Rev. A* **45**, 3403 "
            "(1992). "
            "[doi:10.1103/PhysRevA.45.3403](https://doi.org/10.1103/PhysRevA.45.3403)",
        ),
        (
            "cao1997",
            "L. Cao, “Practical method for determining the minimum "
            "embedding dimension of a scalar time series”, *Physica D* "
            "**110**, 43 (1997). "
            "[doi:10.1016/S0167-2789(97)00118-8](https://doi.org/10.1016/S0167-2789(97)00118-8)",
        ),
    ],
    "Recurrence & RQA": [
        (
            "eckmann1987",
            "J.-P. Eckmann, S. O. Kamphorst & D. Ruelle, “Recurrence plots "
            "of dynamical systems”, *Europhys. Lett.* **4**, 973 (1987). "
            "[doi:10.1209/0295-5075/4/9/004](https://doi.org/10.1209/0295-5075/4/9/004)",
        ),
        (
            "zbilut1992",
            "J. P. Zbilut & C. L. Webber, “Embeddings and delays as derived "
            "from quantification of recurrence plots”, *Phys. Lett. A* "
            "**171**, 199 (1992). "
            "[doi:10.1016/0375-9601(92)90426-M](https://doi.org/10.1016/0375-9601(92)90426-M)",
        ),
        (
            "trulla1996",
            "L. L. Trulla, A. Giuliani, J. P. Zbilut & C. L. Webber, "
            "“Recurrence quantification analysis of the logistic equation "
            "with transients”, *Phys. Lett. A* **223**, 255 (1996). "
            "[doi:10.1016/S0375-9601(96)00741-4](https://doi.org/10.1016/S0375-9601(96)00741-4)",
        ),
        (
            "marwan2007",
            "N. Marwan, M. C. Romano, M. Thiel & J. Kurths, “Recurrence "
            "plots for the analysis of complex systems”, *Phys. Rep.* "
            "**438**, 237 (2007). "
            "[doi:10.1016/j.physrep.2006.11.001](https://doi.org/10.1016/j.physrep.2006.11.001)",
        ),
    ],
    "Fixed points, periodic orbits & interval methods": [
        (
            "schmelcher1997",
            "P. Schmelcher & F. K. Diakonos, “Detecting unstable periodic "
            "orbits of chaotic dynamical systems”, *Phys. Rev. Lett.* **78**, "
            "4733 (1997). "
            "[doi:10.1103/PhysRevLett.78.4733](https://doi.org/10.1103/PhysRevLett.78.4733)",
        ),
        (
            "davidchack1999",
            "R. L. Davidchack & Y.-C. Lai, “Efficient algorithm for "
            "detecting unstable periodic orbits in chaotic systems”, "
            "*Phys. Rev. E* **60**, 6172 (1999). "
            "[doi:10.1103/PhysRevE.60.6172](https://doi.org/10.1103/PhysRevE.60.6172)",
        ),
        (
            "krawczyk1969",
            "R. Krawczyk, “Newton-Algorithmen zur Bestimmung von "
            "Nullstellen mit Fehlerschranken”, *Computing* **4**, 187 "
            "(1969). [doi:10.1007/BF02234767](https://doi.org/10.1007/BF02234767)",
        ),
        (
            "neumaier1990",
            "A. Neumaier, *Interval Methods for Systems of Equations*, "
            "Cambridge University Press (1990).",
        ),
    ],
    "Attractors, basins & global stability": [
        (
            "grebogi1983",
            "C. Grebogi, S. W. McDonald, E. Ott & J. A. Yorke, “Final state "
            "sensitivity: an obstruction to predictability”, *Phys. Lett. "
            "A* **99**, 415 (1983). "
            "[doi:10.1016/0375-9601(83)90945-3](https://doi.org/10.1016/0375-9601(83)90945-3)",
        ),
        (
            "menck2013",
            "P. J. Menck, J. Heitzig, N. Marwan & J. Kurths, “How basin "
            "stability complements the linear-stability paradigm”, *Nat. "
            "Phys.* **9**, 89 (2013). "
            "[doi:10.1038/nphys2516](https://doi.org/10.1038/nphys2516)",
        ),
        (
            "daza2015",
            "A. Daza, A. Wagemakers, M. A. F. Sanjuán & J. A. Yorke, "
            "“Testing for basins of Wada”, *Sci. Rep.* **5**, 16579 "
            "(2015). [doi:10.1038/srep16579](https://doi.org/10.1038/srep16579)",
        ),
        (
            "daza2016",
            "A. Daza, A. Wagemakers, B. Georgeot, D. Guéry-Odelin & M. A. "
            "F. Sanjuán, “Basin entropy: a new tool to analyze "
            "uncertainty in dynamical systems”, *Sci. Rep.* **6**, 31416 "
            "(2016). [doi:10.1038/srep31416](https://doi.org/10.1038/srep31416)",
        ),
        (
            "halekotte2020",
            "L. Halekotte & U. Feudel, “Minimal fatal shocks in multistable "
            "complex networks”, *Sci. Rep.* **10**, 11783 (2020). "
            "[doi:10.1038/s41598-020-68805-6](https://doi.org/10.1038/s41598-020-68805-6)",
        ),
        (
            "datseris2022",
            "G. Datseris & A. Wagemakers, “Effortless estimation of basins "
            "of attraction”, *Chaos* **32**, 023104 (2022). "
            "[doi:10.1063/5.0076568](https://doi.org/10.1063/5.0076568)",
        ),
        (
            "datseris2023",
            "G. Datseris, K. L. Rossi & A. Wagemakers, “Framework for global "
            "stability analysis of dynamical systems”, *Chaos* **33**, "
            "073151 (2023). "
            "[doi:10.1063/5.0159675](https://doi.org/10.1063/5.0159675)",
        ),
    ],
    "Orbit diagrams, Poincaré maps & the classics": [
        (
            "poincare1899",
            "H. Poincaré, *Les méthodes nouvelles de la mécanique "
            "céleste*, Gauthier-Villars (1892–1899).",
        ),
        (
            "may1976",
            "R. M. May, “Simple mathematical models with very complicated "
            "dynamics”, *Nature* **261**, 459 (1976). "
            "[doi:10.1038/261459a0](https://doi.org/10.1038/261459a0)",
        ),
        (
            "feigenbaum1978",
            "M. J. Feigenbaum, “Quantitative universality for a class of "
            "nonlinear transformations”, *J. Stat. Phys.* **19**, 25 "
            "(1978). [doi:10.1007/BF01020332](https://doi.org/10.1007/BF01020332)",
        ),
        (
            "henon1982",
            "M. Hénon, “On the numerical computation of Poincaré "
            "maps”, *Physica D* **5**, 412 (1982). "
            "[doi:10.1016/0167-2789(82)90034-3](https://doi.org/10.1016/0167-2789(82)90034-3)",
        ),
    ],
}


def _render_methods() -> str:
    out: list[str] = []
    for topic, entries in _METHODS.items():
        out.append(f"### {topic}")
        out.append("")
        for _key, line in sorted(entries, key=lambda kv: kv[0]):
            out.append(f"- {line}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

_INTRO = """---
description: The literature behind TSDynamics — the original papers for every built-in system, and the method papers behind the analysis toolkit, with DOI links.
---

<span class="ts-kicker">References · Bibliography</span>

# Bibliography

Every built-in system and every analysis method in TSDynamics traces back to a
primary source. This page collects them all — never a competitor library, always
the original paper. It has two halves:

- The **[systems bibliography](#systems)** is generated straight from the
  catalogue. Each built-in system carries a `reference` (and, where one exists, a
  `doi`) class attribute; this page walks the [registry](../reference/registry.md),
  deduplicates, and lists the citing systems under each paper. Add or edit a
  system and its citation lands here automatically.
- The **[methods bibliography](#methods)** collects the papers behind the
  analysis toolkit — the same citations that close each page in the
  [Analysis](../analysis/index.md) section.

!!! tip "Reproducing the numbers"
    The parameter defaults and initial conditions in each system's page match the
    cited paper wherever the source gives them. When a quantity is quoted in the
    docs (a Lyapunov spectrum, a fractal dimension) it is pinned to a fixed
    initial condition and, for stochastic systems, a fixed seed — so every number
    is reproducible from the snippet that produces it.
"""


def build_page() -> str:
    """Assemble the full bibliography Markdown page (intro + systems + methods)."""
    registry = _load_registry()
    grouped, n_systems, n_papers, n_without = _collect_systems(registry)

    parts: list[str] = [_INTRO.rstrip(), ""]

    # ---- Systems bibliography -------------------------------------------
    parts.append("## Systems {#systems}")
    parts.append("")
    parts.append(
        f"The {n_systems} built-in systems cite **{n_papers} distinct sources**, "
        "grouped below by family and ordered alphabetically by first author. A "
        "handful of textbook/folklore systems (the Lissajous figures, a couple of "
        "purely illustrative maps) carry no single primary source and are omitted "
        "from this list."
    )
    parts.append("")

    for family in _FAMILY_ORDER:
        papers = grouped.get(family)
        if not papers:
            continue
        parts.append(f"### {_FAMILY_LABEL[family]}")
        parts.append("")
        for paper in papers:
            parts.append(_render_systems_paper(paper))
        parts.append("")

    # ---- Methods bibliography -------------------------------------------
    parts.append("## Methods {#methods}")
    parts.append("")
    parts.append(
        "The primary literature behind the analysis toolkit. These are the "
        "citations that close each page in the [Analysis](../analysis/index.md) "
        "section, collected here by topic; every implementation names its source "
        "in its docstring and on its prose page."
    )
    parts.append("")
    parts.append(_render_methods().rstrip())
    parts.append("")

    # ---- Footer ---------------------------------------------------------
    parts.append("---")
    parts.append("")
    parts.append(
        "*The systems bibliography is generated from the registry by "
        "`docs/_tooling/make_bibliography.py`; re-run it after adding or editing a "
        "system. If a DOI resolves to the wrong paper, fix the `doi` class "
        "attribute on the system, not this page.*"
    )
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    """Write (or, with ``--check``, verify) the bibliography page; return an exit code."""
    ap = argparse.ArgumentParser(description="Generate the bibliography page.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and diff against the committed file (CI mode)",
    )
    args = ap.parse_args()

    page = build_page()

    if args.check:
        current = _OUT.read_text(encoding="utf-8") if _OUT.exists() else ""
        if current != page:
            sys.stderr.write(
                f"[make_bibliography] {_OUT.relative_to(_ROOT)} is out of date — "
                "re-run `.venv/bin/python docs/_tooling/make_bibliography.py`.\n"
            )
            return 1
        sys.stdout.write("[make_bibliography] up to date.\n")
        return 0

    _OUT.write_text(page, encoding="utf-8")
    sys.stdout.write(f"[make_bibliography] wrote {_OUT.relative_to(_ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
