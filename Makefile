
.PHONY: lint #         | Run linter
lint:
	  uvx ruff check

.PHONY: format #       | Format code
format:
		uvx ruff format

.PHONY: help #         | List targets
help:                                                                                                                    
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20