---
description: How to cite TSDynamics — its Zenodo DOI and software BibTeX entry, the original papers behind its methods, and the per-system reference and DOI carried by every built-in.
---

<span class="ts-kicker">Project · Citation</span>

# Citation

If TSDynamics contributes to published work, please cite it — and, just as
importantly, cite the **original papers** behind the methods and models you
relied on. The scholarly credit belongs first to the people who invented the
model and the algorithm; the software is the instrument.

## Citing the software

TSDynamics is archived on Zenodo and has a persistent DOI:
[**10.5281/zenodo.20945679**](https://doi.org/10.5281/zenodo.20945679). This is
the *concept* DOI — it always resolves to the newest release, so you never have
to update it. Cite it alongside the exact version you actually used; the
installed version is always available at runtime:

```python
import tsdynamics as ts
ts.__version__          # e.g. '{{ tsdynamics_version }}'
```

A BibTeX entry for the software:

```bibtex
@software{estevez_tsdynamics,
  author  = {Estevez, Daniel},
  title   = {{TSDynamics}: compiled dynamical systems and chaos analysis for {Python}},
  year    = {2026},
  version = {{{ tsdynamics_version }}},
  doi     = {10.5281/zenodo.20945679},
  url     = {https://doi.org/10.5281/zenodo.20945679},
  note    = {MIT license. Set version = to your installed tsdynamics.__version__.}
}
```

!!! note "Cite this repository"
    `10.5281/zenodo.20945679` is the **concept DOI**: it always resolves to the
    newest published release. Zenodo also mints a distinct version DOI for each
    release, so if you need to pin a specific archived version cite that release's
    own DOI instead — but for a citation that never goes stale, use the concept
    DOI above and pin the `version` field to your installed
    `tsdynamics.__version__`.

## Citing the methods

TSDynamics implements published algorithms. Alongside the software entry, cite
the original paper for each method your results depend on:

| You used | Cite |
| -------- | ---- |
| `lyapunov_spectrum` (QR / tangent dynamics) | Benettin, Galgani, Giorgilli & Strelcyn, *Lyapunov characteristic exponents for smooth dynamical systems…*, Meccanica **15**, 9–30 (1980) |
| `lyapunov_spectrum` (two-trajectory rescaling, Jacobian-free fallback) | Benettin, Galgani & Strelcyn, *Kolmogorov entropy and numerical experiments*, Phys. Rev. A **14**, 2338 (1976) |
| `lyapunov_from_data` (Kantz) | Kantz, *A robust method to estimate the maximal Lyapunov exponent of a time series*, Phys. Lett. A **185**, 77 (1994) |
| `lyapunov_from_data` (Rosenstein) | Rosenstein, Collins & De Luca, *A practical method for calculating largest Lyapunov exponents from small data sets*, Physica D **65**, 117 (1993) |
| `kaplan_yorke_dimension` | Kaplan & Yorke, *Chaotic behavior of multidimensional difference equations*, LNM **730**, Springer (1979) |
| `correlation_dimension` / `correlation_sum` | Grassberger & Procaccia, *Characterization of strange attractors*, Phys. Rev. Lett. **50**, 346 (1983) |
| `gali` (chaos indicator) | Skokos, Bountis & Antonopoulos, *Geometrical properties of local dynamics…*, Physica D **231**, 30 (2007) |
| `zero_one_test` | Gottwald & Melbourne, *A new test for chaos in deterministic systems*, Proc. R. Soc. A **460**, 603 (2004) |
| `expansion_entropy` | Hunt & Ott, *Defining chaos*, Chaos **25**, 097618 (2015) |
| `recurrence_matrix` / `rqa` | Marwan, Romano, Thiel & Kurths, *Recurrence plots for the analysis of complex systems*, Phys. Rep. **438**, 237 (2007) |
| `embed` (delay reconstruction) | Takens, *Detecting strange attractors in turbulence*, LNM **898**, Springer (1981) |
| `embedding_dimension` (Cao / FNN) | Cao, *Practical method for determining the minimum embedding dimension…*, Physica D **110**, 43 (1997) |
| `attractors` / `basins` | Datseris & Wagemakers, *Effortless estimation of basins of attraction*, Chaos **32**, 023104 (2022) |
| `basin_entropy` | Daza, Wagemakers, Georgeot, Guéry-Odelin & Sanjuán, *Basin entropy: a new tool to analyze uncertainty in dynamical systems*, Sci. Rep. **6**, 31416 (2016) |

Each analysis page under [Analysis](../analysis/index.md) lists the exact
reference for the routines it documents, and every routine's docstring cites the
paper it implements.

## Citing the systems

Each built-in system declares its literature source in its `_reference` class
attribute — shown on its page under [Systems](../systems/index.md), printed by
`system.info`, and available programmatically from the
[registry](../references/index.md):

```python
from tsdynamics import registry

registry.get("Lorenz").reference
# 'Lorenz (1963), J. Atmos. Sci. 20, 130-141'
```

Most systems also carry the bare DOI of that primary reference, sourced where
available from the published catalogue metadata — the registry entry is the door:

```python
registry.get("Lorenz").doi
# '10.1175/1520-0469(1963)020<0130:dnf>2.0.co;2'
```

Of the 177 built-in systems, 172 declare a literature reference and 155 carry a
DOI; every one of those 155 is rendered as a resolvable `doi.org` link on that
system's generated page. To pull the reference for every system you touched — the makings of a
`\bibliography` — sweep the registry:

```python
from tsdynamics import registry

for entry in registry.all_systems():
    ref = entry.reference or "(no reference on file)"
    print(f"{entry.name:24s} {ref}")
```

If your results hinge on a particular model, cite its original paper. The model
deserves the credit before the implementation does — the Lorenz attractor is
Lorenz's, not ours.
