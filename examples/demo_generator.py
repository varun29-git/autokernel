"""Example command-provider protocol. This is deterministic, not an LLM."""
import hashlib
import json
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


request = json.load(sys.stdin)
source = canonical({"operation": "add"})
task_digest = hashlib.sha256(canonical(request["task"])).hexdigest()
candidate = {
    "schema_version": 1, "task_digest": task_digest, "backend": "numpy_demo", "family": "generated_vector_add",
    "source_digest": hashlib.sha256(source).hexdigest(), "parameters": {"chunk": 8192}, "parent_ids": [],
    "hypothesis": "Use a whole-row NumPy addition to reduce per-chunk dispatch overhead.",
    "expected_saved_ms": 0.0, "workspace_bytes": 0,
}
print(json.dumps({"candidate": candidate, "source": source.decode()}))
