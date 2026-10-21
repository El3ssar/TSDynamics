# External API inventories

These Sphinx inventory snapshots preserve external API cross-links without
requiring network access during a strict documentation build. An HTTP 503 from
the Python documentation server previously caused an otherwise valid CI build
to fail.

Downloaded on 2026-10-21:

| File | Source |
| --- | --- |
| `python.inv` | <https://docs.python.org/3.12/objects.inv> |
| `numpy.inv` | <https://numpy.org/doc/stable/objects.inv> |
| `scipy.inv` | <https://docs.scipy.org/doc/scipy/objects.inv> |

To refresh a snapshot, download its source URL to the corresponding file and
run `TSD_DOCS_FIGURES=0 uv run mkdocs build --strict`. Keep the matching
`base_url` in `mkdocs.yml`; it is the destination for generated links.

The inventories contain reference names and URLs, rather than copies of the
upstream manuals. `hooks/inventories.py` resolves them before mkdocstrings starts
loading inventories.
