# ============================================================================ #
# Spidy — Developer Makefile                                                   #
# Usage: make <target>                                                         #
# Requires: Python 3.11+, pip                                                  #
# ============================================================================ #

.PHONY: install install-dev run test lint format typecheck clean help

## ─── Setup ─────────────────────────────────────────────────────────────────

install:          ## Install runtime dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:      ## Install all dev + runtime dependencies
	pip install -r requirements-dev.txt
	pip install -e .

## ─── Run ────────────────────────────────────────────────────────────────────

run:              ## Start Spidy
	python -m spidy.main

## ─── Testing ────────────────────────────────────────────────────────────────

test:             ## Run all unit tests
	pytest tests/unit/ -v

test-all:         ## Run all tests including integration
	pytest tests/ -v

test-cov:         ## Run tests with coverage report
	pytest tests/unit/ --cov=spidy --cov-report=term-missing

## ─── Code Quality ───────────────────────────────────────────────────────────

lint:             ## Run ruff linter
	ruff check spidy/ tests/

format:           ## Auto-format code with ruff
	ruff format spidy/ tests/

typecheck:        ## Run mypy type checker
	mypy spidy/ --ignore-missing-imports

check:            ## Run all quality checks (lint + format + type)
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) typecheck

## ─── Cleanup ────────────────────────────────────────────────────────────────

clean:            ## Remove build artifacts and caches
	rmdir /s /q __pycache__ 2>nul || true
	rmdir /s /q .pytest_cache 2>nul || true
	rmdir /s /q .mypy_cache 2>nul || true
	rmdir /s /q .ruff_cache 2>nul || true
	rmdir /s /q htmlcov 2>nul || true
	del /q coverage.xml 2>nul || true

## ─── Help ───────────────────────────────────────────────────────────────────

help:             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
