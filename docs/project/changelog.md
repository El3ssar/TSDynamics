---
description: Where TSDynamics release notes live, how versions are numbered, and how the changelog is generated.
---

<span class="ts-kicker">Project · Changelog</span>

# Changelog

Release notes are **generated automatically** from the conventional-commit
history by [python-semantic-release](https://python-semantic-release.readthedocs.io/).
Nobody writes them by hand — every released change is accounted for because the
commit that introduced it carries its own `feat:` / `fix:` / `perf:` prefix.

## Where the notes live

- **[GitHub Releases](https://github.com/El3ssar/TSDynamics/releases)** — the
  canonical, per-tag notes, published the moment a release is cut. From v1.0.0
  onward this is the source of truth.
- **[CHANGELOG.md](https://github.com/El3ssar/TSDynamics/blob/main/CHANGELOG.md)**
  in the repository root — kept up to date by the same tooling; it carries the
  historical record through v1.0.0.

## How versions are numbered

The version bump is decided by the [Conventional Commits](https://www.conventionalcommits.org/)
prefixes since the last release ([semantic versioning](https://semver.org/)):

| Commit type | Bump | Example |
| ----------- | ---- | ------- |
| `feat:` | **minor** — `5.2.6 → 5.3.0` | a new system, analysis, or backend |
| `fix:` / `perf:` | **patch** — `5.2.6 → 5.2.7` | a bug fix or a speed-up |
| `!` / `BREAKING CHANGE:` | **major** — `5.2.6 → 6.0.0` | a change to the public API |
| `docs:` / `chore:` / `ci:` | none | no release is cut |

The installed version is always available at runtime:

```python
import tsdynamics as ts
ts.__version__          # e.g. '5.2.6'
```

## A recent snippet

The most recent milestone in the checked-in
[CHANGELOG.md](https://github.com/El3ssar/TSDynamics/blob/main/CHANGELOG.md),
the 1.0 stabilisation:

```text
## v1.0.0

### Bug Fixes
- base: widen ParamSet.param_hash from 32 to 64 bits
- dde:  explicit _delay_params with validation; LE zero-weight warning
- maps: Jacobian correctness, parameter order, and robust LE loop
- ode:  variable-dim systems compile cleanly; KS large-L is non-trivial

### Features
- base:  support class-level default_ic in SystemBase.resolve_ic

### Refactoring
- systems: make every ODE/DDE _equations keyword-only
- rename the base classes to ContinuousSystem / DelaySystem / DiscreteMap
```

Since v1.0.0 the project moved its entire integration engine to native Rust
(the v3 milestone), grew the [analysis toolkit](../analysis/index.md) and the
[visualization layer](../visualization/index.md), and added the delay and
stochastic families — all reflected in the per-release notes on GitHub. For the
full narrative, read the [GitHub Releases](https://github.com/El3ssar/TSDynamics/releases)
timeline.
