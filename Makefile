# TSDynamics v3 parallel-dev convenience targets (ROADMAP §5/§6).
# These wrap the stream-claiming helper and the xval scaffold; everyday Python
# work still goes through `uv run …` as documented in CONTRIBUTING.md.
.DEFAULT_GOAL := help
.PHONY: help streams claim unclaim worktree xval xval-build \
        test test-slow test-all test-full

help: ## Show these targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── Change-scoped testing (CI-CHANGED) ─────────────────────────────────────────
# The everyday loop. `--changed` runs only the tests your diff vs origin/main can
# affect (foundational changes fall back to the full suite); `-n auto` parallelises.
# Override the diff base with `make test BASE=<ref>` or `TSD_CHANGED_BASE=<ref>`.
# DO NOT run the full suite to check routine work — `make test` is the right loop.

test: ## Change-scoped fast tests for your diff (the everyday loop)
	@uv run pytest --changed $(if $(BASE),--changed-since=$(BASE),) -m "not slow and not full" --no-cov -n auto

test-slow: ## Change-scoped slow tier (long sims) for your diff
	@uv run pytest --changed $(if $(BASE),--changed-since=$(BASE),) -m "slow and not full" --no-cov -n auto

test-all: ## Full fast tier over EVERY system/analysis (parallel) — pre-push sanity
	@uv run pytest -m "not slow and not full" --no-cov -n auto

test-full: ## Fast + slow tiers over everything (parallel); excludes the nightly -m full
	@uv run pytest -m "not full" --no-cov -n auto

streams: ## List free / blocked / taken v3 streams (ROADMAP §6.0)
	@scripts/claim-stream.sh list

claim: ## Claim a stream and re-check the race: make claim ID=E1
	@test -n "$(ID)" || { echo "usage: make claim ID=<stream-id>"; exit 1; }
	@scripts/claim-stream.sh claim $(ID)

unclaim: ## Release a stream back to the pool: make unclaim ID=E1
	@test -n "$(ID)" || { echo "usage: make unclaim ID=<stream-id>"; exit 1; }
	@scripts/claim-stream.sh unclaim $(ID)

worktree: ## Create a stream worktree: make worktree ID=E1 SLUG=interp
	@test -n "$(ID)" || { echo "usage: make worktree ID=<stream-id> [SLUG=<slug>]"; exit 1; }
	@scripts/claim-stream.sh worktree $(ID) $(SLUG)

xval: ## Run the Rust-vs-v2 xval scaffold (Rust path skips if the accelerator is absent)
	@uv run pytest tests/test_xval.py --no-cov -q

xval-build: ## Build+install the v2-seed accelerator, then run the full xval (Rust path included)
	uv run --no-project --with maturin maturin build --release \
		-m crates/tsdynamics-core/Cargo.toml --out crates/tsdynamics-core/dist
	uv pip install --force-reinstall crates/tsdynamics-core/dist/*.whl
	@uv run pytest tests/test_xval.py --no-cov -q
