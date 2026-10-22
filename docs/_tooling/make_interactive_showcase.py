"""Generate real browser players from the animation guide's exact public examples.

Run ``uv run --no-sync python docs/_tooling/make_interactive_showcase.py`` to
refresh both players and their posters; ``--check`` needs only Python's standard
library and rejects stale assets without starting a simulation or a browser.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import make_animation_showcase as movies

ROOT = movies.ROOT
PAGE = movies.PAGE
OUT = ROOT / "docs/assets/interactive"
MANIFEST = OUT / "showcase.json"
ENHANCEMENT = ROOT / "docs/js/interactive-showcase.js"
REGENERATE = "uv run --no-sync python docs/_tooling/make_interactive_showcase.py"
EXAMPLES = {"interactive-orbit": "threejs", "interactive-trace": "plotly"}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rust_sources() -> list[Path]:
    """Read production crate sources, never Cargo's generated target/build trees.

    Keep discovery independent of Git so the freshness guard also works in a
    source archive and notices new source modules before they are staged.
    """
    crates = ROOT / "crates"
    paths = [path for source in crates.glob("tsdyn-*/src") for path in source.rglob("*.rs")]
    paths += list(crates.glob("tsdyn-*/build.rs"))
    return sorted(paths)


def renderer_fingerprint() -> str:
    """Track web drawing semantics, the wrapper and data/model acquisition code."""
    source = ROOT / "src/tsdynamics"
    paths = sorted((source / "viz").rglob("*.py"))
    paths += sorted((source / "viz/render/threejs/_assets").glob("*.js"))
    paths += [source / "systems/continuous/chaotic_attractors.py"]
    paths += sorted((source / "families").rglob("*.py"))
    paths += sorted((source / "_engine").rglob("*.py"))
    paths += _rust_sources()
    paths += [Path(__file__), ENHANCEMENT]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            tree = movies._RendererSemantics().visit(ast.parse(text))
            text = repr(movies._canonical(tree))
        digest.update(text.encode())
    return digest.hexdigest()


def build_plot(slug: str):
    """Execute the printed fence without hidden input or an injected namespace."""
    namespace = {}
    exec(compile(movies.example_source(slug), f"animation.md:{slug}", "exec"), namespace)
    plot = namespace.get("web_plot")
    if plot is None or not plot.is_animated:
        raise ValueError(f"{slug}: expected an animated plot named web_plot")
    return plot


def _plotly_document(document: str, slug: str) -> str:
    """Add a loading/failure still around the otherwise unchanged public export."""
    enhancement = ENHANCEMENT.read_text(encoding="utf-8")
    if "</script" in enhancement.lower():
        raise ValueError("The inline enhancement must not contain a script closing tag.")
    style = """<title>Inspect a Lorenz time trace · TSDynamics</title><style>
html,body {margin:0;height:100%;background:#171923;overflow:hidden}
#tsd-live-poster {position:fixed;inset:0;width:100%;height:100%;object-fit:contain;
  background:#171923;z-index:20;pointer-events:none}
#tsd-live-status {position:fixed;right:12px;bottom:12px;z-index:21;max-width:85%;
  color:#e8eaf0;background:#171923dd;padding:7px 10px;border-radius:6px;
  font:12px system-ui,sans-serif}
[hidden] {display:none!important}
</style>"""
    fallback = (
        f'<img id="tsd-live-poster" src="{slug}-poster.png" '
        'alt="The complete Lorenz x time trace, shown while the live view loads" />'
        '<p id="tsd-live-status" role="status">Loading the interactive view…</p>'
        f"<script>{enhancement}</script>"
    )
    page = document.replace("</head>", style + "</head>", 1).replace(
        "<body>", "<body>" + fallback, 1
    )
    return "\n".join(line.rstrip() for line in page.splitlines()) + "\n"


def render_example(slug: str, backend: str) -> dict:
    """Write actual HTML through Plot.save, plus a matching final-frame poster."""
    source_hash = hashlib.sha256(movies.example_source(slug).encode()).hexdigest()
    fingerprint = renderer_fingerprint()
    plot = build_plot(slug)
    OUT.mkdir(parents=True, exist_ok=True)
    page = OUT / f"{slug}.html"
    poster = OUT / f"{slug}-poster.png"
    options = {"axes": False} if backend == "threejs" else {}
    plot.save(str(page), backend=backend, **options)
    if backend == "plotly":
        page.write_text(_plotly_document(page.read_text(encoding="utf-8"), slug), encoding="utf-8")
    static = plot.copy()
    static.animation = None
    static.save(str(poster), size=(960, 600 if backend == "threejs" else 540), dpi=120)
    if (
        hashlib.sha256(movies.example_source(slug).encode()).hexdigest() != source_hash
        or renderer_fingerprint() != fingerprint
    ):
        raise ValueError(f"Sources changed while rendering {slug}. Run: {REGENERATE}")
    return {
        "backend": backend,
        "source_sha256": source_hash,
        "renderer_sha256": fingerprint,
        "html_sha256": _hash(page),
        "poster_sha256": _hash(poster),
        "forward_seconds": plot.animation.playback_seconds(len(plot.layers[0].data["x"])),
    }


def validate_assets() -> list[str]:
    """Return actionable freshness errors without optional dependencies."""
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or set(manifest["examples"]) != set(EXAMPLES):
            raise ValueError("The interactive showcase manifest is incomplete.")
        fingerprint = renderer_fingerprint()
        errors = []
        for slug, backend in EXAMPLES.items():
            record = manifest["examples"][slug]
            expected = {
                "backend": backend,
                "source_sha256": hashlib.sha256(movies.example_source(slug).encode()).hexdigest(),
                "renderer_sha256": fingerprint,
                "html_sha256": _hash(OUT / f"{slug}.html"),
                "poster_sha256": _hash(OUT / f"{slug}-poster.png"),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                errors.append(f"{slug}: interactive player or its example has changed")
        if errors:
            raise ValueError("\n".join(errors))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [f"Interactive showcase is stale: {exc}\nRun: {REGENERATE}"]
    return []


def main() -> None:
    """Render both live examples, or check the checked-in assets cheaply."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        errors = validate_assets()
        if errors:
            raise SystemExit("\n".join(errors))
        print("Both interactive browser exports are current.")
        return
    import matplotlib

    matplotlib.use("Agg")
    records = {slug: render_example(slug, backend) for slug, backend in EXAMPLES.items()}
    MANIFEST.write_text(
        json.dumps({"schema_version": 1, "examples": records}, indent=2) + "\n", encoding="utf-8"
    )
    print("Rendered both real interactive HTML players and their static posters.")


if __name__ == "__main__":
    main()
