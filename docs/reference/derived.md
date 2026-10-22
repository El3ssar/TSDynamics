---
description: API reference for tsdynamics.derived — PoincareMap, StroboscopicMap, TangentSystem, ProjectedSystem, Ensemble, WrappedSystem.
---

<span class="ts-kicker">Reference</span>

# Derived systems

Wrappers that re-present an existing system through a new lens while keeping the
[`System`](base.md) protocol intact. Analyses work with a derived view when its
family and capabilities meet their requirements, and it still answers `run`. Prose
introduction: [the mental model](../start/concepts.md#derived-systems-composition).

Construct section, stroboscopic and ensemble views from the model:

```python
# skip-doctest — the two verbs, for reference (`sys` is any flow)
sys.poincare("y", 0.0)          # a PoincareMap   — a section plane
sys.poincare(period=4.488)      # a StroboscopicMap — the phase circle
sys.ensemble(states)            # an Ensemble     — many copies, one object
```

Construct tangent and projected systems explicitly
(`ts.derived.TangentSystem(system, k=2)`, `ts.derived.ProjectedSystem(system, (0, 2))`).

::: tsdynamics.derived.poincare.PoincareMap

::: tsdynamics.derived.stroboscopic.StroboscopicMap

::: tsdynamics.derived.tangent.TangentSystem

::: tsdynamics.derived.ensemble.Ensemble

::: tsdynamics.derived.projected.ProjectedSystem

::: tsdynamics.derived.wrapped.WrappedSystem

::: tsdynamics.derived._base.DerivedSystem
