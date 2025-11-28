.PHONY: lint #         | Run linter
lint:
	   uv run ruff check

.PHONY: format #       | Check formatting
format:
		uv run ruff format --diff

.PHONEY: format-fix #   | Fix formatting issues
format-fix:
		uv run ruff format .

.PHONY: typecheck #    | Run type checker
typecheck:
		uv run mypy .

.PHONY: protos #       | Generate protobuf files
protos:
		uv run ./scripts/generate_protos.py

.PHONY: update #       | Update git submodules
update:
		git submodule update --init --recursive

.PHONY: pull #         | Pull and update git submodules
pull:
		git pull
		git submodule update --init --recursive

.PHONY: test #         | Run tests
test:
		uv run pytest -v

.PHONY: reset-sync #   | Reset virtual environment and sync dependencies again
reset-sync:
		rm -rf .venv
		rm uv.lock
		uv sync

.PHONY: init #         | One-time setup: install hooks and generate protos
init:
		$(MAKE) hooks-install
		$(MAKE) protos

.PHONY: hooks-install # | Install pre-commit hooks
hooks-install:
		uv run pre-commit install

.PHONY: hooks-run #    | Run all hooks on all files
hooks-run:
		uv run pre-commit run --all-files

.PHONY: hooks-update # | Update hook versions
hooks-update:
		uv run pre-commit autoupdate

.PHONY: help #         | List targets
help:
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20