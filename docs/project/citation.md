---
description: How to cite TSDynamics, and the original papers behind the methods and systems it implements.
---

<span class="ts-kicker">Project</span>

# Citation

If TSDynamics contributes to published work, cite the software:

```bibtex
@software{estevez_tsdynamics,
  author = {Estevez, Daniel},
  title  = {TSDynamics: Compiled dynamical systems and chaos analysis for Python},
  url    = {https://github.com/El3ssar/TSDynamics},
  note   = {Version as installed; see tsdynamics.__version__},
}
```

---

## Citing the methods

TSDynamics implements published methods. Alongside the software entry,
cite the original papers for what your work uses:

| You used | Cite |
| -------- | ---- |
| `max_lyapunov` (two-trajectory rescaling) | Benettin, Galgani & Strelcyn, *Kolmogorov entropy and numerical experiments*, Phys. Rev. A **14**, 2338 (1976) |
| `lyapunov_spectrum` (QR / tangent dynamics) | Benettin, Galgani, Giorgilli & Strelcyn, *Lyapunov characteristic exponents for smooth dynamical systems...*, Meccanica **15**, 9–30 (1980) |
| `kaplan_yorke_dimension` | Kaplan & Yorke, *Chaotic behavior of multidimensional difference equations*, Lecture Notes in Mathematics **730**, Springer (1979) |

## Citing the systems

Each built-in system declares its literature source in its `reference`
class attribute, shown on its page under [Systems](../systems/index.md)
and available programmatically:

```python
from tsdynamics import registry

registry.get("Lorenz").reference
# 'Lorenz (1963), J. Atmos. Sci. 20, 130-141'
```

If your results hinge on a particular system, cite its original paper —
the model deserves the credit before the implementation does.
