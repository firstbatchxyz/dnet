UV_RUN = uv run --env-file .env

.PHONY: run #         | Run
run:
	$(UV_RUN)

.PHONY: lint #         | Run linter
lint:
	$(UV_RUN) ruff check

.PHONY: format #       | Check formatting
format:
		$(UV_RUN) ruff format --diff

.PHONEY: format-fix #   | Fix formatting issues
format-fix:
		$(UV_RUN) ruff format .

.PHONY: typecheck #   | Run type checker
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

.PHONY: help #         | List targets
help:
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20