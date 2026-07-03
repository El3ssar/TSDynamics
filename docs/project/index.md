---
description: The Project section — how to cite TSDynamics, contribute code, read the changelog, and add a new system or a new integration solver.
---

<span class="ts-kicker">Project · Overview</span>

# Project

TSDynamics is developed in the open at
[El3ssar/TSDynamics](https://github.com/El3ssar/TSDynamics) by
**Daniel Estevez**, under the MIT license. It is a Python library for studying
dynamical systems — ODEs, delay and stochastic differential equations, and
discrete maps — on a native Rust engine, with a full
[analysis toolkit](../analysis/index.md) that composes over every
[built-in or user-defined system](../systems/index.md).

This section is the contributor's and citer's handbook: how to give the software
and its methods credit, how the project is developed and released, and the two
extension paths — a new **system** for the catalogue, and a new **solver** kernel
for the engine.

## In this section

- [**Citation**](citation.md) — the software BibTeX entry, plus the original
  papers behind the methods and every built-in system. Each system carries its
  literature `reference` (and, where known, a `doi`) programmatically.
- [**Contributing**](contributing.md) — dev setup with [uv](https://docs.astral.sh/uv/),
  the quality gates (`ruff`, `mypy --strict`, the two test tiers), change-scoped
  testing, Conventional Commits, and the PR flow.
- [**Changelog**](changelog.md) — release notes, maintained automatically by
  python-semantic-release from the commit history.
- [**Adding a system**](adding-a-system.md) — drop one class into the catalogue;
  the registry, the test sweep, and the docs page follow for free.
- [**Adding a solver**](adding-a-solver.md) — the recipe for contributing a new
  integration kernel to the Rust engine and mirroring it in the Python registry.

## The shape of the project

TSDynamics ships as **one wheel** — the pure-Python `tsdynamics` package plus a
compiled `tsdynamics._rust` `abi3` extension built from a Cargo workspace of
`tsdyn-*` crates. Because the extension is `abi3` (targeting CPython 3.12), one
wheel per platform covers every CPython ≥ 3.12, and installing needs
**no compiler** — only building from source does (see [Install](../start/install.md)).

Two conventions run through everything a contributor touches:

- **The registry is load-bearing.** Every concrete family subclass
  auto-registers at class-definition time, and the bulk test suite, the docs
  generator, and the top-level namespace all iterate over it. A new system needs
  no test file and no docs page — it inherits both.
- **Cite the original literature.** Docstrings and docs pages cite the paper a
  method or model comes from, so the attribution points at the source rather than
  at whichever implementation we consulted.

## Where things happen

| You want to | Go to |
| ----------- | ----- |
| Report a bug or request a feature | [Issue tracker](https://github.com/El3ssar/TSDynamics/issues) |
| Cite the software or a method | [Citation](citation.md) |
| Set up a dev environment | [Contributing → Dev setup](contributing.md#dev-setup) |
| See what changed in a release | [Changelog](changelog.md) · [GitHub Releases](https://github.com/El3ssar/TSDynamics/releases) |
| Contribute a new attractor / map / SDE | [Adding a system](adding-a-system.md) |
| Contribute a new integration method | [Adding a solver](adding-a-solver.md) |
| Read the full API | [Reference](../references/index.md) |
