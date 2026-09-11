---
description: API reference for tsdynamics.derived — PoincareMap, StroboscopicMap, TangentSystem, ProjectedSystem, Ensemble, WrappedSystem.
---

<span class="ts-kicker">Reference</span>

# Derived systems

Wrappers that re-present an existing system through a new lens while keeping the
[`System`](base.md) protocol intact, so every analysis composes with them
transparently — and so you `run` them exactly as you ran the original. Prose
introduction: [the mental model](../start/concepts.md#derived-systems-composition).

Two of them are reached by a **verb on the system**, because they speak the same
vocabulary the class does:

```python
# skip-doctest — the two verbs, for reference (`sys` is any flow)
sys.poincare("y", 0.0)          # a PoincareMap   — a section plane
sys.poincare(period=4.488)      # a StroboscopicMap — the phase circle
sys.ensemble(states)            # an Ensemble     — many copies, one object
```

The other two are Lyapunov and projection machinery rather than everyday verbs,
so they are constructed at their own address
(`ts.derived.TangentSystem(system, k=2)`, `ts.derived.ProjectedSystem(system, (0, 2))`).

::: tsdynamics.derived.poincare.PoincareMap

::: tsdynamics.derived.stroboscopic.StroboscopicMap

::: tsdynamics.derived.tangent.TangentSystem

::: tsdynamics.derived.ensemble.Ensemble

::: tsdynamics.derived.projected.ProjectedSystem

::: tsdynamics.derived.wrapped.WrappedSystem

::: tsdynamics.derived._base.DerivedSystem
