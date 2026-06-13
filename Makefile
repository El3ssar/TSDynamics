# TSDynamics v3 parallel-dev convenience targets (ROADMAP §5/§6).
# These wrap the stream-claiming helper and the xval scaffold; everyday Python
# work still goes through `uv run …` as documented in CONTRIBUTING.md.
.DEFAULT_GOAL := help
.PHONY: help streams claim unclaim worktree xval xval-build

help: ## Show these targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

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
