"""pyunicorn adapter — recurrence quantification analysis.

``pyunicorn`` is a recurrence-network / RQA toolkit; here it contributes the RQA
determinism row (the dedicated comparison for our A-RQA layer). It does no
integration / entropy / dimension / Lyapunov in the shape this benchmark needs,
so those rows stay blank. The embedding + recurrence criterion are passed as
``**kwargs`` to ``RecurrencePlot``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import series

from adapters._base import BaseAdapter


class PyunicornAdapter(BaseAdapter):
    name = "pyunicorn"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import pyunicorn

        return True, getattr(pyunicorn, "__version__", "?"), ""

    def task_rqa_determinism(self, quick: bool) -> Callable[[], float]:
        from pyunicorn.timeseries import RecurrencePlot

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["rqa_n"]])

        def run() -> float:
            rp = RecurrencePlot(
                x,
                dim=s["rqa_embed_dim"],
                tau=s["rqa_embed_delay"],
                recurrence_rate=s["rqa_recurrence_rate"],
                silence_level=2,
            )
            return float(rp.determinism(l_min=2))

        return run
