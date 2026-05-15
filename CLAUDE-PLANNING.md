# Where to start when you open this repo

If you're a Claude (or any contributor) picking up TSDynamics work, **read these
two files first, in this order**:

1. [`.planning/STATUS.md`](.planning/STATUS.md) — current milestone + literal next action.
2. [`.planning/milestones/<NN-name>.md`](.planning/milestones/) — the file STATUS
   points at. Self-contained spec.

That's enough to start working. Read [`.planning/CONTRIBUTING-CLAUDE.md`](.planning/CONTRIBUTING-CLAUDE.md)
for the full protocol (commit conventions, closing-a-chat checklist, what to
leave alone).

For the strategic shape of the project — the full roadmap, the four tracks, why
Rust, why phased — see [`.planning/ROADMAP.md`](.planning/ROADMAP.md) and the
design docs under [`.planning/design/`](.planning/design/).

For codebase conventions (ruff, line length, base-class contract, test
patterns) see [`CLAUDE.md`](CLAUDE.md).

## TL;DR mission

Become the reference dynamical-systems library. Supersede DynamicalSystems.jl.
Pure-Rust compute layer, invisible to users. Easy system definition stays easy.

## Protocol in one paragraph

Open `STATUS.md`. Read the milestone it points to. Ask the user any open
questions in that milestone before writing code. Work in small commits with
tests. Before closing the chat, tick the milestone in `ROADMAP.md`, rewrite
`STATUS.md` with the next-action and any new open questions, commit `.planning/`
alongside the code.
