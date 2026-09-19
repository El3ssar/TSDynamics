# TSDynamics convenience targets.
# Everyday Python work goes through `uv run …` as documented in CONTRIBUTING.md.
.DEFAULT_GOAL := help
.PHONY: help test test-slow test-all test-full

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
