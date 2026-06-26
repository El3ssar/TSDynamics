---
description: API reference for tsdynamics.registry — programmatic discovery of every registered system class.
---

<span class="ts-kicker">Reference</span>

# Registry

Runtime registry of system classes. Every concrete subclass of a family
base auto-registers at class-definition time — the 151 built-ins and your
own classes alike. The registry is what the bulk test suite and the
documentation generator iterate over.

```python
from tsdynamics import registry

registry.families()                       # {'ode': 120, 'dde': 5, 'map': 26}
registry.categories(family="map")         # {'chaotic_maps': 9, ...}
entry = registry.get("Lorenz")            # SystemEntry
entry.cls, entry.family, entry.category   # (<class Lorenz>, 'ode', 'chaotic_attractors')
```

::: tsdynamics.registry
