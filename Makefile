.PHONY: lint #         | Run linter
lint:
	  uvx ruff check

.PHONY: format #       | Check formatting
format:
		uvx ruff format --diff

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

.PHONY: help #         | List targets
help:                                                                                                                    
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20