.PHONY: mdns #         | Show dns-sd services
mdns:
		dns-sd -Q _dnet_p2p._tcp.local. PTR

.PHONY: lint #         | Run linter
lint:
	  uvx ruff check

.PHONY: format #       | Format code
format:
		uvx ruff format

.PHONY: protos #       | Generate protobuf files
protos:
		uv run ./srcipts/generate_protos.py

#### MODEL LOADERS #####
.PHONY: prep-qwen3 # | Prepare & Load Qwen3 model on shards
prep-qwen3:
		export TOPOLOGY=$(curl -s -X POST http://localhost:8080/v1/prepare_topology -H \"Content-Type: application/json\" -d '{ \"model\": \"Qwen/Qwen3-4B-MLX-4bit\" }'); \
		curl -X POST http://localhost:8080/v1/load_model -d "$$TOPOLOGY"



.PHONY: help #         | List targets
help:                                                                                                                    
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20