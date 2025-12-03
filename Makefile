UV_RUN = uv run --env-file .env

.PHONY: run #          | Run
run:
	$(UV_RUN)

.PHONY: lint #         | Run linter
lint:
	$(UV_RUN) ruff check

.PHONY: lint-fix #     | Fix linting issues
lint-fix:
	   uv run ruff check --fix

.PHONY: format #       | Check formatting
format:
		$(UV_RUN) ruff format --diff

.PHONEY: format-fix #   | Fix formatting issues
format-fix:
		$(UV_RUN) ruff format .

.PHONY: typecheck #    | Run type checker
typecheck:
		$(UV_RUN) mypy .

.PHONY: protos #       | Generate protobuf files
protos:
		$(UV_RUN) ./scripts/generate_protos.py

.PHONY: update #       | Update git submodules
update:
		git submodule update --init --recursive

.PHONY: pull #         | Pull and update git submodules
pull:
		git pull
		git submodule update --init --recursive

.PHONY: test #         | Run tests
test:
		$(UV_RUN) pytest -v

.PHONY: reset-sync #   | Reset virtual environment and sync dependencies again
reset-sync:
		rm -rf .venv
		rm uv.lock
		uv sync

.PHONY: init #         | One-time setup: install hooks and generate protos
init:
		$(MAKE) ensure-env
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

.PHONY: ensure-env #   | Ensure .env exists
ensure-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo 'Copied .env.example to .env'; \
	else \
		echo '.env already exists'; \
	fi

.PHONY: help #         | List targets
help:
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20