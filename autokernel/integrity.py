from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def atomic_write(path, content: bytes, *, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, canonical(value) + b"\n")


def snapshot(root, *, include=(".py", ".json", ".toml", ".cu", ".cuh", ".h", ".hpp")):
    root = Path(root).resolve()
    excluded = {".git", ".venv", "__pycache__", "artifacts", "build", "dist", ".pytest_cache"}
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in excluded or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix not in include:
            continue
        if path.is_symlink():
            raise ValueError(f"source snapshot refuses symlink: {relative}")
        if path.is_file():
            files[str(relative)] = file_digest(path)
    if not files:
        raise ValueError("empty source snapshot")
    return {"files": files, "digest": digest(files)}
