# Build lifecycle for arbiter.
#
# `make` lists the targets. `make pre-commit` is the gate that must pass
# before anything is committed: formatting, lints, tests and docs, with
# warnings treated as errors.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Every target is a verb, not a file.
.PHONY: help build release test test-unit test-doc lint fmt fmt-check clippy \
        doc check check-embed run cti-mcq cti-mcq-run cti-mcq-render \
        datasets clean pre-commit ci

# Lints are errors in CI and in the pre-commit gate, warnings interactively.
CLIPPY_FLAGS ?= --all-targets --all-features
CARGO ?= cargo

help: ## List the available targets
	@echo "arbiter — make targets"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Binaries land in \$$CARGO_TARGET_DIR, not ./target."

## ---------------------------------------------------------------- build

build: ## Compile the library, binary and tests (debug)
	$(CARGO) build --all-features --all-targets

release: ## Compile optimised
	$(CARGO) build --release --all-features

## ---------------------------------------------------------------- test

test: ## Run every test: unit, integration and doc
	$(CARGO) test --all-features

test-unit: ## Run only the unit and integration tests
	$(CARGO) test --all-features --lib --tests

test-doc: ## Run only the documentation tests
	$(CARGO) test --all-features --doc

## ---------------------------------------------------------------- style

fmt: ## Rewrite the source in canonical style
	$(CARGO) fmt --all

fmt-check: ## Fail if any source is not canonically formatted
	$(CARGO) fmt --all --check

clippy: ## Lint, treating every warning as an error
	$(CARGO) clippy $(CLIPPY_FLAGS) -- -D warnings

lint: fmt-check clippy check-embed ## Check formatting and lints without changing anything

check: ## Type-check without producing binaries
	$(CARGO) check --all-features --all-targets

check-embed: ## Prove the library still compiles as a bare embeddable tool
	# No features, library only: this is what a consumer embedding the tool
	# gets. It must not need an argument parser, a subscriber, or a runtime.
	$(CARGO) check --no-default-features --lib

doc: ## Build the API documentation, failing on a broken intra-doc link
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --all-features --no-deps

## ---------------------------------------------------------------- gate

pre-commit: fmt lint test doc ## Format, then prove the tree is correct and clean
	@echo
	@echo "pre-commit: formatting, lints, tests and docs are clean."

ci: lint test doc ## The pre-commit gate without rewriting any file
	@echo
	@echo "ci: formatting, lints, tests and docs are clean."

## ---------------------------------------------------------------- run

run: ## Ask a sample question (needs TYPESAFE_API_KEY)
	$(CARGO) run --quiet --features cli -- choice "Which team should own this ticket?" \
		--option 'payments=Checkout, billing, or payment processing issues.' \
		--option 'frontend=Rendering, layout, or browser compatibility issues.' \
		--option 'account=Login, permissions, or profile issues.' \
		--state "My checkout page shows a blank screen after I click Pay." \
		--min-confidence 0.75

cti-mcq: cti-mcq-run cti-mcq-render ## Ask the CTI questions, then draw the page (needs a key)

cti-mcq-run: ## Ask the CTI questions and record every attempt (needs a key; costs money)
	$(CARGO) run --quiet --release --example cti_mcq -- run

cti-mcq-render: ## Draw the HTML page from the recorded run (free, no network)
	$(CARGO) run --quiet --release --example choice_report

datasets: ## Fetch the CTI-MCQ split into ./cti-mcq.tsv (needs network)
	$(CARGO) run --quiet --release --example cti_mcq -- fetch

clean: ## Remove build artefacts
	$(CARGO) clean
