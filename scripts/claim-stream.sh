#!/usr/bin/env bash
# claim-stream.sh — the v3 parallel-dev claiming helper (ROADMAP §6.0).
#
# There is no shared runtime memory between concurrent sessions; the only state
# every session can observe is git + GitHub. This script automates the §6.0
# protocol against that state: list which streams are free & unblocked, then
# claim one (assign + comment + re-check the race window before coding).
#
# The live claim state lives in GitHub issues, NOT in any checked-in file. This
# is tooling around that source of truth, not a second copy of it.
#
# Usage:
#   scripts/claim-stream.sh list                 # free/blocked/taken streams (default)
#   scripts/claim-stream.sh claim <ID|#num>      # assign @me + comment + race re-check
#   scripts/claim-stream.sh worktree <ID> [slug] # git worktree add for a claimed stream
#   scripts/claim-stream.sh unclaim <ID|#num>    # release back to the pool
#
# Requires: gh (authenticated), git, python3.
set -euo pipefail

LABEL="stream"
WORKTREE_DIR=".claude/worktrees"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_PY="${HERE}/stream_status.py"

die() {
    echo "error: $*" >&2
    exit 1
}
need() { command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"; }
need gh
need git
need python3

# All stream issues (open + closed) as JSON: number, title, state, assignees, body.
_issues_json() {
    gh issue list --label "$LABEL" --state all --limit 200 \
        --json number,title,state,assignees,body
}

# Open PR head refs + remote stream branches, one per line (2nd "taken" signal).
_inflight_refs() {
    gh pr list --state open --limit 200 --json headRefName \
        --jq '.[].headRefName' 2>/dev/null || true
    git ls-remote --heads origin 'refs/heads/stream/*' 2>/dev/null \
        | sed 's#.*refs/heads/##' || true
}

# Resolve a stream ID (e.g. F4) or "#23"/"23" to an issue number.
_resolve_number() {
    local arg="$1"
    if [[ "$arg" =~ ^#?([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    _issues_json | python3 "$STATUS_PY" resolve "$arg"
}

cmd_list() {
    local refs
    refs="$(_inflight_refs)"
    _issues_json | REFS="$refs" python3 "$STATUS_PY" list
}

cmd_claim() {
    [[ $# -ge 1 ]] || die "usage: claim-stream.sh claim <ID|#num>"
    local num
    num="$(_resolve_number "$1")"
    local me
    me="$(gh api user --jq .login)"

    # Pre-claim race check: bail if already assigned.
    local pre
    pre="$(gh issue view "$num" --json assignees --jq '[.assignees[].login]|join(",")')"
    if [[ -n "$pre" ]]; then
        die "#$num is already assigned to: $pre — pick another (run: list)"
    fi

    echo "Claiming #$num as @$me ..."
    gh issue edit "$num" --add-assignee "@me" >/dev/null
    gh issue comment "$num" \
        --body "Claimed by session **$me** via \`scripts/claim-stream.sh\` (ROADMAP §6.0)." >/dev/null

    # §6.0 step 4: re-check once. If someone else also landed in the gap, release.
    sleep 2
    local post
    post="$(gh issue view "$num" --json assignees --jq '[.assignees[].login]|join(",")')"
    if [[ "$post" != "$me" ]]; then
        echo "race detected (assignees now: ${post:-none}); releasing my claim." >&2
        gh issue edit "$num" --remove-assignee "@me" >/dev/null || true
        die "could not secure #$num — re-run: list"
    fi
    local id
    id="$(gh issue view "$num" --json title --jq .title | sed -E 's/^\[([^]]+)\].*/\1/')"
    echo "Secured #$num [$id]. Next:"
    echo "  scripts/claim-stream.sh worktree $id <slug>"
    echo "  # then open ONE PR titled '<type>: [$id] …' with 'Closes #$num' in the body."
}

cmd_worktree() {
    [[ $# -ge 1 ]] || die "usage: claim-stream.sh worktree <ID> [slug]"
    local id="$1" slug="${2:-}"
    [[ -n "$slug" ]] || slug="$(echo "$id" | tr '[:upper:]' '[:lower:]')"
    local branch="stream/${id}-${slug}"
    local path="${WORKTREE_DIR}/tsd-${id}"
    git worktree add "$path" -b "$branch" main
    echo "worktree ready at $path (branch $branch)"
}

cmd_unclaim() {
    [[ $# -ge 1 ]] || die "usage: claim-stream.sh unclaim <ID|#num>"
    local num
    num="$(_resolve_number "$1")"
    gh issue edit "$num" --remove-assignee "@me" >/dev/null
    gh issue comment "$num" \
        --body "Unassigned — returning this stream to the pool (ROADMAP §6.0)." >/dev/null
    echo "Released #$num back to the pool."
}

case "${1:-list}" in
    list) cmd_list ;;
    claim)
        shift
        cmd_claim "$@"
        ;;
    worktree)
        shift
        cmd_worktree "$@"
        ;;
    unclaim)
        shift
        cmd_unclaim "$@"
        ;;
    -h | --help | help) sed -n '2,20p' "$0" ;;
    *) die "unknown command '$1' (try: list | claim | worktree | unclaim)" ;;
esac
