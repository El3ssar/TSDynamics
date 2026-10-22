---
description: The reference shelf for TSDynamics — cross-library benchmarks, the literature behind every method and system, and the complete auto-generated API.
---

<span class="ts-kicker">References</span>

# References

Use this section to look up benchmark methods, original literature and API
documentation. For a guided introduction, begin with [Start](../start/index.md)
or the [Analysis guides](../analysis/index.md).

## Benchmarks

[**Benchmarks &rarr;**](benchmarks.md)

A head-to-head comparison against the established Python dynamical-systems
libraries on a shared set of classic tasks — short and long ODE integration,
integration accuracy, the Lyapunov family, correlation dimension, the
bifurcation sweep, basins, fixed points, the Poincaré section, and a spread of
signal-based complexity measures. It records task-specific **timings** and,
where a comparison value is available, the estimate and its deviation. Results
apply to the stated hardware, inputs and settings; a literature comparison is
a cross-check, not a general accuracy guarantee.

## Bibliography

[**Bibliography &rarr;**](bibliography.md)

The original literature, in two parts. The **methods** half cites the papers
behind the analysis toolkit — the Lyapunov, dimension, recurrence, embedding
and basin routines each trace back to a specific source, never to
another library. The **systems** half is the catalogue's provenance: every
built-in system carries the paper that defined it (and, where one exists, its
DOI), pulled straight from the [registry](../reference/registry.md) so the page
cannot drift from the code.

## API reference

[**API reference &rarr;**](../reference/index.md)

The complete public API, one page per area, rendered from the docstrings in the
source:

- [Top level](../reference/top-level.md) — everything importable straight from `tsdynamics`.
- [Base classes](../reference/base.md) — `ContinuousSystem`, `DelaySystem`, `DiscreteMap`, `StochasticSystem`, `Trajectory`, and the `System` protocol.
- [Derived systems](../reference/derived.md) — `PoincareMap`, `StroboscopicMap`, `TangentSystem`, `ProjectedSystem`, `Ensemble`, `WrappedSystem`.
- [Analysis](../reference/analysis.md) — the full quantifier toolkit.
- [Data &amp; state-space](../reference/data.md) — `Trajectory`, regions, samplers, set distances.
- [Solvers](../reference/solvers.md) — the by-name method registry and auto-stiffness.
- [Registry](../reference/registry.md) — programmatic discovery of the built-in catalogue.
- [Utilities](../reference/utils.md) — the shared output-grid helper.

The [visualization layer](../visualization/index.md) has its own dedicated
section, and each built-in system is documented on its own generated page under
[Systems](../systems/index.md).
