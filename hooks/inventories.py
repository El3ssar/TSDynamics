"""Resolve external API cross-references from checked-in inventory snapshots.

An upstream documentation outage must not fail an otherwise valid strict build.
The original base URLs still determine where the generated links point.
"""

from pathlib import Path

from mkdocs.plugins import event_priority


@event_priority(100)
def on_config(config):
    """Use local inventories while retaining their public documentation URLs."""
    directory = Path(__file__).resolve().parents[1] / "docs" / "_tooling" / "inventories"
    handler = config["plugins"]["mkdocstrings"].config["handlers"]["python"]
    for inventory in handler["inventories"]:
        if not inventory["url"].startswith("file:"):
            inventory["url"] = (directory / inventory["url"]).as_uri()
    return config
