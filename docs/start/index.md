---
description: Install TSDynamics, run a first trajectory, and learn the mental model the rest of the documentation builds on.
---

<span class="ts-kicker">Start</span>

# Start here

Three pages: installation, a complete worked example covering all three
system families, and the concepts everything else builds on. They are the
only prerequisites for the rest of the documentation.

<div class="grid cards" markdown>

- **[Install](install.md)**

    ---

    Installation with pip or uv, the C compiler requirement for
    JIT-compiled integration, and the optional extras.

- **[First trajectory](first-trajectory.md)**

    ---

    A complete worked example: integrate the Lorenz system, compute its
    Lyapunov spectrum, iterate the Hénon map, and integrate a delay
    equation with a history function.

- **[The mental model](concepts.md)**

    ---

    A system is a params dict, a dimension, and one method. The three
    subclass contracts, the compile-once cache, the `System` protocol,
    and the registry.

</div>
