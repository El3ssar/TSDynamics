#!/usr/bin/env bash
# Bootstrap one GitHub issue per ticket from a program's tickets.tsv (the claim board).
#
#   scripts/bootstrap-issues.sh [path/to/tickets.tsv] [--create]
#
#   scripts/bootstrap-issues.sh                                   # DRY RUN, default TSV
#   scripts/bootstrap-issues.sh planning/production/tickets.tsv   # DRY RUN, explicit TSV
#   scripts/bootstrap-issues.sh planning/production/tickets.tsv --create   # create
#
# Generalizes scripts/bootstrap-v4-issues.sh to the 13-column dispatch schema
# (id title track tier effort slug phase depends owns acceptance mode care verify).
# Back-compat: a 10-column legacy row (the old streams.tsv) loads with
# mode=A care=routine verify="". Idempotent (skips an existing "[ID] title").
# Live claim state = these issues; the human board is the program ROADMAP.
# Protocol: docs/contributing/agent-dispatch-playbook.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYBOOK="docs/contributing/agent-dispatch-playbook.md"

TSV="$ROOT/planning/polish-design/streams.tsv"   # default (legacy v4 board)
CREATE=0
for arg in "$@"; do
  case "$arg" in
    --create) CREATE=1 ;;
    *) TSV="$arg" ;;
  esac
done
# allow a repo-relative TSV path
[[ -f "$TSV" ]] || TSV="$ROOT/$TSV"

[[ -f "$TSV" ]] || { echo "missing TSV: $TSV"; exit 1; }
command -v gh >/dev/null || { echo "gh CLI required"; exit 1; }

ensure_label() { gh label create "$1" --color "$2" --description "$3" 2>/dev/null || true; }

if [[ $CREATE -eq 1 ]]; then
  ensure_label stream 0e8a16 "A dispatchable work ticket"
  for t in 1 2 3 4 5; do ensure_label "tier:$t" 5319e7 "Dependency tier $t (lower = more foundational)"; done
  for tr in STRUCT IFACE PERF VIZREADY POLISH MIGRATE \
            CORRECTNESS FINISH-V4 VIZ-GROUND DOCS-IA DOCS-CONTENT DOCS-ENG EXAMPLES PROD-HARDEN RELEASE; do
    ensure_label "track:$tr" 1d76db "Work track $tr"
  done
  for ph in P0 P1 P2 P3 P4 P5; do ensure_label "phase:$ph" fbca04 "Program phase $ph"; done
  ensure_label "mode:A" 0e8a16 "Dispatch mode A — parallel Workflow fan-out"
  ensure_label "mode:B" b60205 "Dispatch mode B — dedicated single session (high care)"
  ensure_label "care:routine"     c2e0c6 "Author + one adversarial pass"
  ensure_label "care:adversarial" fbca04 "Skeptic subagent + slow-tier regression"
  ensure_label "care:security"    b60205 "Adds /security-review + threat-model note"
fi

existing="$(gh issue list --state all --limit 1000 --json title --jq '.[].title' 2>/dev/null || true)"

# Fields: id title track tier effort slug phase depends owns acceptance [mode] [care] [verify]
# NB: parse with a manual tab-split, NOT `IFS=$'\t' read` — tab is an IFS-whitespace
# char, so read would COLLAPSE empty fields (e.g. an empty `depends`) and shift columns.
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  F=(); rest="$line"
  while [[ "$rest" == *$'\t'* ]]; do F+=("${rest%%$'\t'*}"); rest="${rest#*$'\t'}"; done
  F+=("$rest")
  id="${F[0]:-}"; title="${F[1]:-}"; track="${F[2]:-}"; tier="${F[3]:-}"; effort="${F[4]:-}"
  slug="${F[5]:-}"; phase="${F[6]:-}"; depends="${F[7]:-}"; owns="${F[8]:-}"; acceptance="${F[9]:-}"
  mode="${F[10]:-}"; care="${F[11]:-}"; verify="${F[12]:-}"
  [[ -z "$id" || "$id" == "id" ]] && continue
  # legacy 10-column rows: default the dispatch metadata
  [[ -z "$mode"   ]] && mode="A"
  [[ -z "$care"   ]] && care="routine"

  issue_title="[$id] $title"
  if grep -Fxq "$issue_title" <<<"$existing"; then echo "skip (exists): $issue_title"; continue; fi

  deps_md="none"; [[ -n "$depends" ]] && deps_md="$depends"
  owns_list="- ${owns// ; /$'\n'- }"
  bf="$(mktemp)"
  {
    printf 'Ticket **%s** — phase `%s`, track `%s`, tier `%s`, effort `%s`.\n\n' "$id" "$phase" "$track" "$tier" "$effort"
    printf '%s\n\n' "$title"
    printf '**Mode:** `%s` · **Care:** `%s`\n\n' "$mode" "$care"
    printf '**Depends on:** %s  ·  **Branch:** `stream/%s-%s`  ·  **Worktree:** `.claude/worktrees/tsd-%s`\n\n' "$deps_md" "$id" "$slug" "$id"
    printf '**Owns:**\n%s\n\n' "$owns_list"
    printf '**Acceptance:**\n%s\n\n' "$acceptance"
    [[ -n "$verify" ]] && printf '**Verify (beyond `make test`):**\n%s\n\n' "$verify"
    printf -- '---\n'
    printf 'Dispatch protocol: [`%s`](../blob/main/%s). Claim, then ' "$PLAYBOOK" "$PLAYBOOK"
    printf '`git fetch origin && git worktree add .claude/worktrees/tsd-%s -b stream/%s-%s origin/main`; ' "$id" "$id" "$slug"
    printf 'one PR titled `<type>: [%s] …` with `Closes #<this issue>`; the maintainer merges.\n' "$id"
  } > "$bf"

  if [[ $CREATE -eq 1 ]]; then
    url="$(gh issue create --title "$issue_title" --body-file "$bf" \
        --label stream --label "tier:$tier" --label "track:$track" --label "phase:$phase" \
        --label "mode:$mode" --label "care:$care")"
    echo "created: $issue_title -> $url"
  else
    echo "would create: $issue_title  (mode:$mode care:$care track:$track phase:$phase tier:$tier)"
  fi
  rm -f "$bf"
done < <(tail -n +2 "$TSV")

echo
[[ $CREATE -eq 1 ]] && echo "Done." || echo "DRY RUN — re-run with --create to make the issues."
