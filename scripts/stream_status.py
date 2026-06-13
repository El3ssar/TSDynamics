#!/usr/bin/env python3
"""Analyze the v3 stream board from ``gh`` issue JSON (helper for ``claim-stream.sh``).

Reads the JSON emitted by ``gh issue list --label stream --state all --json
number,title,state,assignees,body`` on stdin and classifies each stream per the
ROADMAP §6.0 protocol. Two modes:

* ``list``    — print FREE-&-UNBLOCKED / BLOCKED / TAKEN / DONE buckets. Reads the
  ``REFS`` env var (newline-separated open PR head refs + remote ``stream/*``
  branches) as the second "taken" signal.
* ``resolve <ID>`` — print the issue number for a stream ID (e.g. ``F4``).

This is read-only analysis; the mutating steps (assign/comment/worktree) live in
``claim-stream.sh``. The live claim state is GitHub itself — this only reports it.
"""

from __future__ import annotations

import json
import os
import re
import sys

_ID_RE = re.compile(r"\[([^\]]+)\]")
# Stream-ID-shaped tokens: a letter run, optional ``-WORD`` segments, optional
# trailing digits — matches F0, E7, E-SDE, A-BASIN, C-DERIV, I-WHEEL, ...
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*[0-9]*")


def _sid(issue: dict) -> str:
    """Return the stream ID from an issue title ``[<ID>] <name>`` (upper-cased)."""
    m = _ID_RE.match(issue["title"])
    return (m.group(1).strip() if m else issue["title"]).upper()


def _deps(issue: dict, known: set[str]) -> list[str]:
    """Return the stream IDs this issue declares as blockers (``Depends on:`` lines)."""
    out: set[str] = set()
    self_id = _sid(issue)
    for line in issue["body"].splitlines():
        if re.search(r"depends on", line, re.IGNORECASE):
            for tok in _TOKEN_RE.findall(line):
                t = tok.upper()
                if t in known and t != self_id:
                    out.add(t)
    return sorted(out)


def _taken_by_ref(stream_id: str, refs: list[str]) -> bool:
    """Whether an open PR / remote branch ``stream/<id>-*`` exists for this stream."""
    prefix = f"stream/{stream_id.lower()}-"
    return any(r.lower().startswith(prefix) for r in refs)


def cmd_resolve(arg: str) -> int:
    """Print the issue number for stream ID ``arg`` (or error out)."""
    want = arg.strip().upper()
    for issue in json.load(sys.stdin):
        if _sid(issue) == want:
            print(issue["number"])
            return 0
    sys.stderr.write(f"no stream issue with ID {arg}\n")
    return 1


def cmd_list() -> int:
    """Print the classified stream board to stdout."""
    issues = json.load(sys.stdin)
    refs = [r for r in os.environ.get("REFS", "").splitlines() if r.strip()]
    known = {_sid(it) for it in issues}
    state = {_sid(it): it["state"].upper() for it in issues}

    free: list[tuple[str, int, list[str]]] = []
    blocked: list[tuple[str, int, list[str]]] = []
    taken: list[tuple[str, int, str]] = []
    done: list[tuple[str, int]] = []

    for it in issues:
        sid = _sid(it)
        if it["state"].upper() == "CLOSED":
            done.append((sid, it["number"]))
            continue
        assignees = [a["login"] for a in it["assignees"]]
        deps = _deps(it, known)
        open_blockers = [b for b in deps if state.get(b) != "CLOSED"]
        if assignees or _taken_by_ref(sid, refs):
            taken.append((sid, it["number"], ", ".join(assignees) or "branch/PR in flight"))
        elif open_blockers:
            blocked.append((sid, it["number"], open_blockers))
        else:
            free.append((sid, it["number"], deps))

    def row(sid: str, num: int, extra: str = "") -> str:
        return f"  #{num:<3} {sid:<9} {extra}"

    print("── FREE & UNBLOCKED (claimable now) " + "─" * 22)
    if free:
        for sid, num, deps in sorted(free):
            print(row(sid, num, f"(deps ok: {', '.join(deps) or 'none'})"))
    else:
        print("  (none — every unblocked stream is taken)")

    print("\n── BLOCKED (dependency issues still open) " + "─" * 16)
    for sid, num, ob in sorted(blocked):
        print(row(sid, num, f"waiting on: {', '.join(ob)}"))

    print("\n── TAKEN (assigned or branch/PR in flight) " + "─" * 15)
    for sid, num, who in sorted(taken):
        print(row(sid, num, f"by {who}"))

    print("\n── DONE (issue closed / merged) " + "─" * 26)
    print("  " + (", ".join(f"{sid}(#{num})" for sid, num in sorted(done)) or "(none)"))

    if free:
        print(f"\nNext: scripts/claim-stream.sh claim {sorted(free)[0][0]}")
    return 0


def main() -> int:
    """Dispatch on argv[1] (``list`` default, or ``resolve <ID>``)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"
    if mode == "list":
        return cmd_list()
    if mode == "resolve":
        if len(sys.argv) < 3:
            sys.stderr.write("usage: stream_status.py resolve <ID>\n")
            return 2
        return cmd_resolve(sys.argv[2])
    sys.stderr.write(f"unknown mode {mode!r}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
