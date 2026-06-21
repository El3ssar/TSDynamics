// =============================================================================
// Mode-A dispatch — ultracode Workflow parallel fan-out
// =============================================================================
// Clears a BATCH of routine, file-DISJOINT tickets in one session: each ticket
// is implemented by a worktree-isolated subagent, then an independent skeptic
// verifies the diff. The orchestrator (the session that launches this) selects
// the batch, then — for each PASS — claims the issue, pushes the branch, and
// opens ONE PR (left OPEN; the maintainer merges).
//
// USAGE (from the orchestrator session):
//   1. `make streams` → choose FREE ∧ UNBLOCKED ∧ mode=A ∧ care=routine tickets
//      whose `owns` globs are pairwise DISJOINT (see the playbook §2).
//   2. Workflow({ scriptPath: "scripts/workflows/dispatch-batch.mjs",
//                 args: ["DOCS-LYAPUNOV", "DOCS-CHAOS", "FIX-FIXEDMASS-DIGAMMA"] })
//   3. Read the returned reports; for each PASS, claim + push + PR (playbook §2.3).
//
// Ticket specs are read by the agents themselves from
// planning/production/tickets.tsv (grep by id) and/or `gh issue view`.
// See docs/contributing/agent-dispatch-playbook.md for the full protocol.
// =============================================================================

export const meta = {
  name: 'dispatch-batch',
  description: 'Mode-A: implement + adversarially verify a file-disjoint batch of tickets',
  phases: [
    { title: 'Implement', detail: 'one worktree-isolated subagent per ticket' },
    { title: 'Verify', detail: 'an independent skeptic refutes each diff' },
  ],
}

const GUARDRAILS = `
GUARDRAILS (non-negotiable — see docs/contributing/agent-dispatch-playbook.md §4):
- Branch off origin/main, NEVER stale local main: 'git fetch origin' first.
- Edit ONLY files matching the ticket's 'owns' globs (incl. the test file). Touching
  anything else is an automatic FAIL you must self-report.
- NEVER weaken, delete, skip, or xfail an existing assertion, or loosen a numeric
  tolerance, to make a test pass (unless the ticket's acceptance explicitly sanctions a
  named carve-out).
- A correctness fix MUST add a failing-first regression test that asserts the
  literature/known value and fails on the pre-fix code.
- Follow CLAUDE.md: ruff format/lint, NumPy docstrings, NO Julia/.jl references,
  one-file-per-thing. Update CLAUDE.md/docs in the same change if behavior/names change.
- Inner loop is 'make test' (scoped --changed), NOT the full 'uv run pytest'.
- Rust speed: this is a Mode-A (non-Rust) ticket, so you may symlink the prebuilt
  _rust*.so into the worktree and set PYTHONPATH=<worktree>/src to skip a maturin rebuild.
`

const TICKET_CONTEXT = `
You are working ONE ticket of the TSDynamics production+docs program. Repo root:
/home/elessar/Projects/TSDynamics. Read CLAUDE.md (architecture) and the dispatch playbook
(docs/contributing/agent-dispatch-playbook.md). Read YOUR ticket's full spec from
planning/production/tickets.tsv (the row whose first tab-separated field == your TICKET ID):
columns are id,title,track,tier,effort,slug,phase,depends,owns,acceptance,mode,care,verify.
Also run 'gh issue view' for the matching '[ID] ...' issue if it exists.
${GUARDRAILS}
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    slug: { type: 'string' },
    worktree: { type: 'string', description: 'absolute path of your worktree (output of `git rev-parse --show-toplevel`)' },
    branch: { type: 'string', description: 'stream/<id>-<slug>' },
    files_changed: { type: 'array', items: { type: 'string' } },
    owns_respected: { type: 'boolean', description: 'true iff every changed file matches the ticket owns globs' },
    tests_command: { type: 'string' },
    tests_passed: { type: 'boolean' },
    acceptance_self_check: { type: 'string', description: 'each acceptance bullet → met/not-met + evidence' },
    notes: { type: 'string' },
  },
  required: ['id', 'branch', 'worktree', 'files_changed', 'owns_respected', 'tests_passed', 'acceptance_self_check'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    branch: { type: 'string' },
    worktree: { type: 'string' },
    verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    acceptance_checklist: { type: 'string', description: 'each acceptance criterion → pass/fail with evidence from the diff/tests' },
    weakened_assertions_found: { type: 'boolean' },
    verify_results: { type: 'string', description: 'results of running the ticket verify= checks' },
    test_tail: { type: 'string', description: 'last ~15 lines of make test' },
    findings: { type: 'string', description: 'why FAIL, or what was confirmed for PASS' },
    pr_ready: { type: 'boolean' },
  },
  required: ['id', 'branch', 'worktree', 'verdict', 'acceptance_checklist', 'weakened_assertions_found', 'pr_ready'],
}

const ids = Array.isArray(args) ? args.filter(Boolean) : []
if (ids.length === 0) {
  log('No ticket ids passed in args. Pass args: ["TICKET-ID", ...] (a file-disjoint Mode-A batch).')
  return { error: 'no ticket ids', hint: 'Workflow({ scriptPath, args: ["ID1","ID2"] })' }
}
log(`Mode-A batch: ${ids.join(', ')}`)

phase('Implement')
const reports = await pipeline(
  ids,
  // STAGE 1 — implement in an isolated worktree
  (id) =>
    agent(
      `${TICKET_CONTEXT}\n\n=== YOUR TICKET ID: ${id} ===\n` +
        `You are in a fresh git worktree (isolation). Steps: (1) 'git fetch origin'; ` +
        `(2) create your branch 'git checkout -b stream/${id.toLowerCase()}-<slug> origin/main' ` +
        `(<slug> = the ticket's slug column); (3) implement the ticket strictly within its 'owns' ` +
        `globs; (4) run 'make test' (scoped); (5) commit with a Conventional-Commit message ` +
        `'<type>: [${id}] <summary>'; (6) report your worktree absolute path (git rev-parse ` +
        `--show-toplevel), branch, the exact files you changed, whether every changed file is ` +
        `within 'owns', and a per-bullet acceptance self-check. Do NOT push and do NOT open a PR.`,
      { label: `impl:${id}`, phase: 'Implement', schema: IMPL_SCHEMA, isolation: 'worktree', effort: 'high' }
    ),
  // STAGE 2 — an INDEPENDENT skeptic reads the diff in the implementer's worktree and tries to refute it
  (impl, id) => {
    if (!impl || !impl.worktree || !impl.branch) {
      return { id, branch: '(none)', worktree: '(none)', verdict: 'FAIL', acceptance_checklist: 'implementer produced no usable worktree', weakened_assertions_found: false, pr_ready: false, findings: 'Stage 1 returned nothing usable.' }
    }
    return agent(
      `${TICKET_CONTEXT}\n\n=== ADVERSARIALLY VERIFY TICKET ${id} ===\n` +
        `An implementer claims to have completed this ticket in the worktree at:\n  ${impl.worktree}\n` +
        `on branch ${impl.branch}. You did NOT write this code — your job is to try to REFUTE it.\n` +
        `Do: (1) 'git -C ${impl.worktree} diff origin/main' — read the FULL diff; ` +
        `(2) confirm every changed file matches the ticket's 'owns' globs (else FAIL); ` +
        `(3) grep the diff for removed 'assert', loosened atol/rtol, added xfail/skip → if any ` +
        `and not sanctioned by acceptance, FAIL; ` +
        `(4) run the ticket's verify= checks and 'cd ${impl.worktree} && make test' (scoped); ` +
        `(5) check EACH acceptance criterion against the actual diff/tests, not the claim; ` +
        `(6) for a correctness ticket, confirm the regression test fails on origin/main and passes here. ` +
        `Return PASS only if every criterion holds and no guardrail is violated; else FAIL with specifics.`,
      { label: `verify:${id}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' }
    )
  }
)

const clean = reports.filter(Boolean)
const passed = clean.filter((r) => r.verdict === 'PASS' && r.pr_ready)
const failed = clean.filter((r) => !(r.verdict === 'PASS' && r.pr_ready))
log(`PASS: ${passed.map((r) => r.id).join(', ') || 'none'}`)
log(`FAIL/escalate: ${failed.map((r) => r.id).join(', ') || 'none'}`)

// The orchestrator (the launching session) now, for each PASS: claim-stream.sh claim <id>,
// push <worktree> branch, gh pr create (<type>: [id] …, Closes #n, report body) — left OPEN.
// For each FAIL: comment the findings on the issue, unclaim, escalate to Mode B.
return {
  batch: ids,
  pass: passed,
  escalate: failed,
  next_steps: 'For each PASS: claim → git -C <worktree> push -u origin <branch> → gh pr create (leave OPEN). For each escalate: comment findings, unclaim, run in Mode B.',
}
