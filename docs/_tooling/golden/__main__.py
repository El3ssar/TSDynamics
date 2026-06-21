"""Regenerate the golden docs-figure corpus.

Usage (from the repo root, with the ``docs`` dependency group active so that
Matplotlib is importable). Either form works::

    uv run --group docs python docs/_tooling/golden/__main__.py
    uv run --group docs python -m docs._tooling.golden   # if docs/ is a package

Writes :file:`docs/_tooling/golden/figures/<System>.png` for every system in
``GOLDEN_SUBSET`` and rewrites :file:`docs/_tooling/golden/manifest.json` with
their perceptual hashes.  Review the regenerated PNGs by eye and commit them
when an intentional change made them drift.
"""

from __future__ import annotations

import pathlib
import sys

# Make ``golden`` importable whether run as ``python -m docs._tooling.golden``
# (relative import works) or as a bare script (the package dir is on the path).
if __package__:
    from . import GOLDEN_SUBSET, regenerate
else:  # pragma: no cover — exercised only by the bare-script invocation
    # Put the *parent* of this package dir on the path and import the package
    # (the dir name) so the relative-free ``__init__`` loads as ``golden``.
    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent))
    from golden import GOLDEN_SUBSET, regenerate  # type: ignore[no-redef]


def main() -> None:
    """Regenerate the corpus and print a one-line-per-figure summary."""
    manifest = regenerate()
    print(f"regenerated golden corpus: {len(manifest['figures'])} figures")
    for name in GOLDEN_SUBSET:
        info = manifest["figures"][name]
        print(f"  {name:<14} {info['family']:<4} ahash={info['ahash']}")


if __name__ == "__main__":
    main()
