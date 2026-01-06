#!/usr/bin/env python3
"""Generate protobuf files for dnet."""

import glob
import os
import re
from pathlib import Path

from grpc_tools import protoc


def get_pb2_module_name(proto_filename: str) -> str:
    base = Path(proto_filename).stem
    return f"{base}_pb2"


def generate_protos() -> None:
    PROTO_DIR = "src/dnet/protos"
    OUT_DIR = "src/dnet/protos"

    os.makedirs(OUT_DIR, exist_ok=True)

    for proto_file in glob.glob(os.path.join(PROTO_DIR, "*.proto")):
        print(f"Generating proto for {proto_file}")

        ret = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{PROTO_DIR}",
                f"--python_out={OUT_DIR}",
                f"--grpc_python_out={OUT_DIR}",
                f"--mypy_out={OUT_DIR}",
                f"--mypy_grpc_out={OUT_DIR}",
                proto_file,
            ]
        )

        if ret != 0:
            raise RuntimeError(f"protoc failed for {proto_file}")

        # Fix imports in grpc file
        pb2 = get_pb2_module_name(proto_file)
        grpc_file = f"{OUT_DIR}/{pb2}_grpc.py"

        with open(grpc_file, "r+") as f:
            content = f.read()
            content = content.replace(f"import {pb2}", f"from . import {pb2}")
            f.seek(0)
            f.write(content)
            f.truncate()

        print(f"Fixed imports in {grpc_file}")

    # Fix cross-proto imports in all pb2 files
    # (e.g., import dnet_cp_pb2 -> from . import dnet_cp_pb2)
    for pb2_file in glob.glob(os.path.join(OUT_DIR, "*_pb2.py")):
        with open(pb2_file, "r+") as f:
            content = f.read()
            # Match bare imports like "import foo_pb2 as foo__pb2"
            # and convert to relative imports
            pattern = r"^import (\w+_pb2) as (\w+)$"
            replacement = r"from . import \1 as \2"
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            if new_content != content:
                f.seek(0)
                f.write(new_content)
                f.truncate()
                print(f"Fixed cross-proto imports in {pb2_file}")


if __name__ == "__main__":
    generate_protos()
