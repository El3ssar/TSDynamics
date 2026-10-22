"""Render scientific recipe SVGs from the exact public-API fences in recipes.md.

Run ``uv run --no-sync python docs/_tooling/make_viz_recipes.py`` to regenerate
all seven figures, or pass ``--only recipe-data recipe-analysis``.  Rendering
keeps Matplotlib settings and the active library theme local to each call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "visualization" / "recipes.md"
OUT = ROOT / "docs" / "assets" / "figures" / "viz"
RECIPES = (
    "recipe-data",
    "recipe-compare",
    "recipe-linked-views",
    "recipe-ensemble",
    "recipe-analysis",
    "recipe-fields",
    "recipe-spatial",
)
MANIFEST = OUT / "recipes-manifest.json"


def source(slug: str) -> str:
    """Read a recipe's complete marked Python fence; missing code is an error."""
    if slug not in RECIPES:
        raise ValueError(f"Unknown recipe {slug!r}; choose from {RECIPES}.")
    match = re.search(
        rf"<!-- tsd-recipe: {re.escape(slug)} -->\s*```python\n(.*?)```",
        PAGE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Missing marked example {slug!r} in {PAGE}.")
    return match.group(1)


def build(slug: str):
    """Execute the documented recipe and return its public Plot object."""
    namespace = {}
    exec(compile(source(slug), f"{PAGE}:{slug}", "exec"), namespace)
    return namespace["figure"]


def source_hash(slug: str) -> str:
    """Hash the exact runnable fence, normalized to platform-independent newlines."""
    return hashlib.sha256(source(slug).encode("utf-8")).hexdigest()


def _generator_hash() -> str:
    return hashlib.sha256(pathlib.Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()


def validate_assets() -> list[str]:
    """List actionable freshness errors; imports and validation need no plot backend."""
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return ["Recipe manifest missing or invalid. " + _REGENERATE]
    if not isinstance(manifest, dict) or not isinstance(manifest.get("recipes"), dict):
        return ["Recipe manifest has an invalid shape. " + _REGENERATE]
    if manifest.get("generator_sha256") != _generator_hash():
        return ["Recipe renderer changed; regenerate its assets. " + _REGENERATE]
    stale = []
    for slug in RECIPES:
        record = manifest.get("recipes", {}).get(slug, {})
        output = OUT / f"{slug}.svg"
        if (
            record.get("source_sha256") != source_hash(slug)
            or record.get("generator_sha256") != _generator_hash()
            or not output.is_file()
            or record.get("svg_sha256") != hashlib.sha256(output.read_bytes()).hexdigest()
        ):
            stale.append(slug)
    if stale:
        return [f"Recipe assets are stale: {', '.join(stale)}. " + _REGENERATE]
    return []


_REGENERATE = "Run uv run --no-sync python docs/_tooling/make_viz_recipes.py."


def render(slug: str, out_dir: pathlib.Path = OUT) -> pathlib.Path:
    """Save one actual recipe as a reproducible, opaque SVG without global styling."""
    import matplotlib as mpl

    import tsdynamics as ts

    previous_theme = ts.viz.themes.get()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{slug}.svg"
    try:
        with mpl.rc_context(
            {"svg.fonttype": "none", "svg.hashsalt": slug, "savefig.transparent": False}
        ):
            figure = build(slug)
            rendered = figure.render("matplotlib")
            try:
                rendered.savefig(output, metadata={"Date": None}, transparent=False)
            finally:
                # The renderer returns a bare Figure; pyplot.close alone does not free it.
                rendered.clear()
    finally:
        ts.viz.themes.use(previous_theme)
    return output


def main() -> None:
    """Check freshness without rendering, or regenerate all/selected recipes."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only", nargs="+", choices=RECIPES)
    group.add_argument(
        "--check", action="store_true", help="Check media freshness without rendering."
    )
    args = parser.parse_args()
    if args.check:
        errors = validate_assets()
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"All {len(RECIPES)} recipe SVGs match their published source.")
        return
    records = {}
    if MANIFEST.is_file():
        records.update(json.loads(MANIFEST.read_text(encoding="utf-8")).get("recipes", {}))
    for slug in args.only or RECIPES:
        output = render(slug)
        records[slug] = {
            "source_sha256": source_hash(slug),
            "generator_sha256": _generator_hash(),
            "svg_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        print(f"{slug}: {output.stat().st_size:,} bytes → {output}")
    MANIFEST.write_text(
        json.dumps(
            {"version": 1, "generator_sha256": _generator_hash(), "recipes": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
