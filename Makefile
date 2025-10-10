.PHONY: mdns #         | Show dns-sd services
mdns:
		dns-sd -Q _dnet_p2p._tcp.local. PTR

.PHONY: lint #         | Run linter
lint:
	  uvx ruff check

.PHONY: format #       | Check formatting
format:
		uvx ruff format --diff

.PHONY: protos #       | Generate protobuf files
protos:
		uv run ./scripts/generate_protos.py

.PHONY: update #         | Update git submodules
update:
		git submodule update --init --recursive

.PHONY: help #         | List targets
help:                                                                                                                    
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20